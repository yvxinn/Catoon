"""
SoftMaskFusion - Soft Mask 融合

使用软掩码进行归一化加权融合。
Base = sum(M_c' * S_{j_c}) / (sum(M_c') + eps)
"""

import cv2
import numpy as np

from ..context import RoutingPlan, SegmentationOutput, StyleCandidate


class SoftMaskFusion:
    """Soft Mask 融合器"""
    
    def __init__(self, blur_kernel: int = 21):
        """
        初始化融合器
        
        Args:
            blur_kernel: 高斯模糊核大小（用于生成 soft mask）
        """
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
        执行 soft mask 融合
        
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
        
        # 初始化累加器
        weighted_sum = np.zeros((h, w, c), dtype=np.float32)
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)
        
        # 遍历每个区域
        for region_name, config in routing.region_configs.items():
            # 获取该区域的 mask
            mask = seg_out.semantic_masks.get(region_name)
            if mask is None:
                continue
            
            # 生成 soft mask
            soft_mask = self._make_soft_mask(mask)
            
            # 应用权重
            soft_mask = soft_mask * config.mix_weight
            
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
            
            # 扩展 mask 维度用于广播
            mask_3d = soft_mask[:, :, np.newaxis]
            
            # 累加
            weighted_sum += mask_3d * styled_image
            weight_sum += mask_3d
        
        # 归一化
        fused = weighted_sum / (weight_sum + self.eps)
        
        # 处理人脸保护区域
        if routing.face_protection_mask is not None:
            fused = self._apply_face_protection(
                fused, candidates, routing, original_image
            )
        
        # Clamp 到 [0, 1]
        fused = np.clip(fused, 0, 1)
        
        return fused
    
    def _make_soft_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        将硬掩码转换为软掩码
        
        Args:
            mask: 输入掩码 (H, W)
        
        Returns:
            软掩码 (H, W)
        """
        # 确保核大小为奇数
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 应用高斯模糊
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
        """
        应用人脸保护
        
        Args:
            fused: 融合后的图像
            candidates: 候选字典
            routing: 路由计划
            original_image: 原图，用于 strength 混合
        
        Returns:
            应用人脸保护后的图像
        """
        face_mask = routing.face_protection_mask
        if face_mask is None:
            return fused
        
        # 获取人脸区域使用的风格
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


def create_soft_mask_fusion(blur_kernel: int = 21) -> SoftMaskFusion:
    """
    便捷函数：创建 soft mask 融合器
    
    Args:
        blur_kernel: 模糊核大小
    
    Returns:
        SoftMaskFusion 实例
    """
    return SoftMaskFusion(blur_kernel)

