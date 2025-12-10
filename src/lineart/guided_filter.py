"""
GuidedFilter - 导向滤波细节注入

使用 Guided Filter 进行边缘感知的细节增强。
包含降级方案：当 opencv-contrib 不可用时 fallback 到 bilateral。
"""

import cv2
import numpy as np


# 检查 ximgproc 是否可用
_XIMGPROC_AVAILABLE = False
try:
    import cv2.ximgproc
    _XIMGPROC_AVAILABLE = True
except (ImportError, AttributeError):
    _XIMGPROC_AVAILABLE = False


class GuidedFilterEnhancer:
    """Guided Filter 细节增强器"""
    
    def __init__(
        self,
        radius: int = 8,
        eps: float = 0.02,
        detail_strength: float = 0.5,
        fallback_method: str = "bilateral"
    ):
        """
        初始化 Guided Filter 增强器
        
        Args:
            radius: 滤波半径
            eps: 正则化参数（较小值保留更多边缘）
            detail_strength: 细节增强强度
            fallback_method: 降级方法 "bilateral" | "none"
        """
        self.radius = radius
        self.eps = eps
        self.detail_strength = detail_strength
        self.fallback_method = fallback_method
        
        # 检查可用性
        self.use_guided = _XIMGPROC_AVAILABLE
        if not self.use_guided:
            print(f"[GuidedFilter] cv2.ximgproc not available, using fallback: {fallback_method}")
    
    @staticmethod
    def is_available() -> bool:
        """检查 Guided Filter 是否可用"""
        return _XIMGPROC_AVAILABLE
    
    def enhance(
        self,
        image: np.ndarray,
        guide: np.ndarray | None = None
    ) -> np.ndarray:
        """
        使用 Guided Filter 进行细节增强
        
        Args:
            image: 输入图像 float32 (H,W,3) [0,1]
            guide: 引导图像，默认使用原图
        
        Returns:
            增强后的图像 float32
        """
        if guide is None:
            guide = image
        
        if self.use_guided:
            return self._guided_filter_enhance(image, guide)
        elif self.fallback_method == "bilateral":
            return self._bilateral_enhance(image)
        else:
            return image
    
    def _guided_filter_enhance(
        self,
        image: np.ndarray,
        guide: np.ndarray
    ) -> np.ndarray:
        """使用 Guided Filter 进行增强"""
        # 转换为 uint8
        image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        guide_u8 = (np.clip(guide, 0, 1) * 255).astype(np.uint8)
        
        # 如果是彩色图像，分别处理每个通道
        if len(image_u8.shape) == 3:
            # 转换引导图为灰度
            if len(guide_u8.shape) == 3:
                guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
            else:
                guide_gray = guide_u8
            
            # 应用 Guided Filter
            filtered = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=image_u8,
                radius=self.radius,
                eps=self.eps * 255 * 255  # eps 需要根据像素范围调整
            )
        else:
            guide_gray = guide_u8 if len(guide_u8.shape) == 2 else cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
            filtered = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=image_u8,
                radius=self.radius,
                eps=self.eps * 255 * 255
            )
        
        # 转回 float32
        filtered_f32 = filtered.astype(np.float32) / 255.0
        
        # 提取细节层并增强
        # detail = original - smoothed
        # enhanced = original + strength * detail
        detail = image - filtered_f32
        enhanced = image + self.detail_strength * detail
        
        return np.clip(enhanced, 0, 1).astype(np.float32)
    
    def _bilateral_enhance(self, image: np.ndarray) -> np.ndarray:
        """使用 Bilateral Filter 作为降级方案"""
        image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # 使用 bilateral filter 进行平滑
        smoothed = cv2.bilateralFilter(
            image_u8,
            d=self.radius * 2 + 1,
            sigmaColor=75,
            sigmaSpace=75
        )
        
        smoothed_f32 = smoothed.astype(np.float32) / 255.0
        
        # 提取细节并增强
        detail = image - smoothed_f32
        enhanced = image + self.detail_strength * detail
        
        return np.clip(enhanced, 0, 1).astype(np.float32)
    
    def extract_detail_layer(
        self,
        image: np.ndarray,
        guide: np.ndarray | None = None
    ) -> np.ndarray:
        """
        提取细节层（可用于报告展示）
        
        Args:
            image: 输入图像
            guide: 引导图像
        
        Returns:
            细节层 float32
        """
        if guide is None:
            guide = image
        
        # 获取平滑结果
        image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        if self.use_guided:
            guide_u8 = (np.clip(guide, 0, 1) * 255).astype(np.uint8)
            if len(guide_u8.shape) == 3:
                guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
            else:
                guide_gray = guide_u8
            
            smoothed = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=image_u8,
                radius=self.radius,
                eps=self.eps * 255 * 255
            )
        else:
            smoothed = cv2.bilateralFilter(
                image_u8,
                d=self.radius * 2 + 1,
                sigmaColor=75,
                sigmaSpace=75
            )
        
        smoothed_f32 = smoothed.astype(np.float32) / 255.0
        
        # 细节层 = 原图 - 平滑
        detail = image - smoothed_f32
        
        # 归一化到可视范围
        detail = detail * 0.5 + 0.5  # [-1,1] -> [0,1]
        
        return np.clip(detail, 0, 1).astype(np.float32)


def create_guided_filter_enhancer(
    radius: int = 8,
    eps: float = 0.02,
    detail_strength: float = 0.5,
    fallback_method: str = "bilateral"
) -> GuidedFilterEnhancer:
    """
    便捷函数：创建 Guided Filter 增强器
    
    Args:
        radius: 滤波半径
        eps: 正则化参数
        detail_strength: 细节增强强度
        fallback_method: 降级方法
    
    Returns:
        GuidedFilterEnhancer 实例
    """
    return GuidedFilterEnhancer(radius, eps, detail_strength, fallback_method)

