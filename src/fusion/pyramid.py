"""
LaplacianPyramidFusion - Laplacian 金字塔融合

使用多频段融合实现更自然的区域过渡，显著减少接缝伪影。
"""

import cv2
import numpy as np

from ..context import RoutingPlan, SegmentationOutput, StyleCandidate


class LaplacianPyramidFusion:
    """Laplacian 金字塔融合器"""
    
    def __init__(self, levels: int = 6, blur_kernel: int = 21):
        """
        初始化融合器
        
        Args:
            levels: 金字塔层数
            blur_kernel: mask 模糊核大小
        """
        self.levels = levels
        self.blur_kernel = blur_kernel
        self.eps = 1e-8
    
    def fuse(
        self,
        candidates: dict[str, StyleCandidate],
        routing: RoutingPlan,
        seg_out: SegmentationOutput,
        original_image: np.ndarray | None = None,
        region_candidates: dict[str, StyleCandidate] | None = None
    ) -> np.ndarray:
        """
        执行 Laplacian 金字塔融合
        
        Args:
            candidates: {style_id: StyleCandidate} 全局候选字典
            routing: 路由计划
            seg_out: 分割输出
            original_image: 原图 float32 (H,W,3)，用于 strength 混合
            region_candidates: {region_name: StyleCandidate} 区域级候选（优先使用）
        
        Returns:
            融合后的图像 float32 (H,W,3) [0,1]
        """
        # 获取图像尺寸
        first_candidate = next(iter(candidates.values()))
        h, w, c = first_candidate.image.shape
        
        # 确保尺寸适合金字塔操作（需要能被 2^levels 整除）
        pad_h = (2 ** self.levels - h % (2 ** self.levels)) % (2 ** self.levels)
        pad_w = (2 ** self.levels - w % (2 ** self.levels)) % (2 ** self.levels)
        
        # 收集所有需要融合的图像和对应的权重 mask
        images_to_blend = []
        masks_to_blend = []
        
        for region_name, config in routing.region_configs.items():
            # 获取该区域的 mask
            mask = seg_out.semantic_masks.get(region_name)
            if mask is None or mask.sum() < 1:
                continue
            
            # 优先使用区域级候选
            candidate = None
            if region_candidates and region_name in region_candidates:
                candidate = region_candidates[region_name]
            else:
                # 回退到全局候选
                style_id = config.style_id
                candidate = candidates.get(style_id)
                if candidate is None:
                    candidate = candidates.get("Traditional")
            
            if candidate is None:
                continue
            
            # 获取风格化图像
            styled_image = candidate.image
            
            # 应用 strength 参数：混合原图和风格化图像
            if original_image is not None and config.strength < 1.0:
                strength = config.strength
                styled_image = original_image * (1 - strength) + styled_image * strength
            
            # 生成 soft mask
            soft_mask = self._make_soft_mask(mask) * config.mix_weight
            
            # 添加到列表
            images_to_blend.append(styled_image)
            masks_to_blend.append(soft_mask)
        
        if len(images_to_blend) == 0:
            # 没有可融合的图像，返回第一个候选
            return first_candidate.image
        
        if len(images_to_blend) == 1:
            # 只有一个区域，直接返回
            return images_to_blend[0]
        
        # 填充图像和 mask
        images_padded = []
        masks_padded = []
        for img, msk in zip(images_to_blend, masks_to_blend):
            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                msk = np.pad(msk, ((0, pad_h), (0, pad_w)), mode='reflect')
            images_padded.append(img)
            masks_padded.append(msk)
        
        # 执行多图像金字塔融合
        fused = self._multiband_blend_multiple(images_padded, masks_padded)
        
        # 移除填充
        if pad_h > 0 or pad_w > 0:
            fused = fused[:h, :w, :]
        
        # 处理人脸保护区域
        if routing.face_protection_mask is not None:
            fused = self._apply_face_protection(fused, candidates, routing, original_image)
        
        return np.clip(fused, 0, 1).astype(np.float32)
    
    def _multiband_blend_multiple(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray]
    ) -> np.ndarray:
        """
        多图像多频段融合
        
        Args:
            images: 图像列表
            masks: 对应的权重 mask 列表
        
        Returns:
            融合后的图像
        """
        n_images = len(images)
        
        # 构建所有图像的 Laplacian 金字塔
        laplacian_pyramids = []
        for img in images:
            lp = self._build_laplacian_pyramid(img)
            laplacian_pyramids.append(lp)
        
        # 构建所有 mask 的 Gaussian 金字塔
        gaussian_pyramids = []
        for msk in masks:
            # mask 需要扩展为 3 通道
            msk_3ch = np.stack([msk] * 3, axis=-1)
            gp = self._build_gaussian_pyramid(msk_3ch)
            gaussian_pyramids.append(gp)
        
        # 在每一层进行加权融合
        blended_pyramid = []
        for level in range(self.levels):
            # 归一化权重
            weight_sum = sum(gp[level] for gp in gaussian_pyramids) + self.eps
            
            # 加权融合这一层
            blended_level = np.zeros_like(laplacian_pyramids[0][level])
            for i in range(n_images):
                normalized_weight = gaussian_pyramids[i][level] / weight_sum
                blended_level += laplacian_pyramids[i][level] * normalized_weight
            
            blended_pyramid.append(blended_level)
        
        # 重建图像
        fused = self._reconstruct_from_laplacian(blended_pyramid)
        
        return fused
    
    def _build_gaussian_pyramid(self, img: np.ndarray) -> list[np.ndarray]:
        """构建 Gaussian 金字塔"""
        pyramid = [img.astype(np.float32)]
        current = img.astype(np.float32)
        
        for _ in range(self.levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    def _build_laplacian_pyramid(self, img: np.ndarray) -> list[np.ndarray]:
        """构建 Laplacian 金字塔"""
        gaussian = self._build_gaussian_pyramid(img)
        laplacian = []
        
        for i in range(self.levels - 1):
            size = (gaussian[i].shape[1], gaussian[i].shape[0])
            expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
            lap = gaussian[i] - expanded
            laplacian.append(lap)
        
        # 最后一层是 Gaussian 的最顶层
        laplacian.append(gaussian[-1])
        
        return laplacian
    
    def _reconstruct_from_laplacian(self, pyramid: list[np.ndarray]) -> np.ndarray:
        """从 Laplacian 金字塔重建图像"""
        current = pyramid[-1]
        
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = current + pyramid[i]
        
        return current
    
    def _make_soft_mask(self, mask: np.ndarray) -> np.ndarray:
        """将硬掩码转换为软掩码"""
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        soft_mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (kernel_size, kernel_size),
            0
        )
        
        return soft_mask
    
    def _apply_face_protection(
        self,
        fused: np.ndarray,
        candidates: dict[str, StyleCandidate],
        routing: RoutingPlan,
        original_image: np.ndarray | None = None
    ) -> np.ndarray:
        """应用人脸保护"""
        face_mask = routing.face_protection_mask
        if face_mask is None:
            return fused
        
        person_config = routing.region_configs.get("PERSON")
        if person_config is None:
            return fused
        
        face_candidate = candidates.get(person_config.style_id)
        if face_candidate is None:
            face_candidate = candidates.get("Traditional")
        if face_candidate is None:
            return fused
        
        # 获取风格化图像
        face_styled = face_candidate.image
        
        # 应用 strength 参数
        if original_image is not None and person_config.strength < 1.0:
            strength = person_config.strength
            face_styled = original_image * (1 - strength) + face_styled * strength
        
        # 生成 soft face mask
        soft_face_mask = self._make_soft_mask(face_mask)
        mask_3d = soft_face_mask[:, :, np.newaxis]
        
        # 在人脸区域混合
        fused = fused * (1 - mask_3d) + face_styled * mask_3d
        
        return fused


def create_pyramid_fusion(levels: int = 6, blur_kernel: int = 21) -> LaplacianPyramidFusion:
    """
    便捷函数：创建金字塔融合器
    
    Args:
        levels: 金字塔层数
        blur_kernel: 模糊核大小
    
    Returns:
        LaplacianPyramidFusion 实例
    """
    return LaplacianPyramidFusion(levels, blur_kernel)

