"""
CannyLineart - Canny 线稿生成器

使用 Canny 边缘检测生成卡通线稿。
"""

import cv2
import numpy as np


class CannyLineart:
    """Canny 线稿生成器"""
    
    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        line_width: int = 1
    ):
        """
        初始化 Canny 线稿生成器
        
        Args:
            low_threshold: Canny 低阈值
            high_threshold: Canny 高阈值
            line_width: 线条宽度
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.line_width = line_width
    
    def extract(self, image_u8: np.ndarray) -> np.ndarray:
        """
        提取 Canny 边缘
        
        Args:
            image_u8: 输入图像 uint8 (H,W,3)
        
        Returns:
            边缘图 float32 (H,W) [0,1]
        """
        # 转为灰度
        if len(image_u8.shape) == 3:
            gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_u8
        
        # 轻微模糊去噪
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny 边缘检测
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # 调整线条宽度
        # 线宽支持浮点，取整后再膨胀
        int_width = max(1, int(round(self.line_width)))
        if int_width > 1:
            kernel = np.ones((int_width, int_width), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 转为 float32 [0, 1]
        edges = edges.astype(np.float32) / 255.0
        
        return edges
    
    def overlay(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        将边缘叠加到图像（multiply 模式）
        
        Args:
            image: 输入图像 float32 (H,W,3)
            edges: 边缘图 float32 (H,W)
            strength: 线稿强度 0~1
        
        Returns:
            叠加后的图像 float32
        """
        if strength < 0.01:
            return image
        
        # 扩展边缘维度
        edges_3d = edges[:, :, np.newaxis]
        
        # Multiply 模式: output = image * (1 - edges * strength)
        result = image * (1 - edges_3d * strength)
        
        return np.clip(result, 0, 1)
    
    def extract_with_params(
        self,
        image_u8: np.ndarray,
        low: int | None = None,
        high: int | None = None,
        width: int | None = None
    ) -> np.ndarray:
        """
        使用自定义参数提取边缘
        
        Args:
            image_u8: 输入图像
            low: 低阈值
            high: 高阈值
            width: 线条宽度
        
        Returns:
            边缘图
        """
        # 保存原始参数
        orig_low = self.low_threshold
        orig_high = self.high_threshold
        orig_width = self.line_width
        
        # 临时设置
        if low is not None:
            self.low_threshold = low
        if high is not None:
            self.high_threshold = high
        if width is not None:
            self.line_width = width
        
        # 提取
        result = self.extract(image_u8)
        
        # 恢复
        self.low_threshold = orig_low
        self.high_threshold = orig_high
        self.line_width = orig_width
        
        return result


def create_canny_lineart(
    low_threshold: int = 100,
    high_threshold: int = 200,
    line_width: int = 1
) -> CannyLineart:
    """
    便捷函数：创建 Canny 线稿生成器
    
    Args:
        low_threshold: 低阈值
        high_threshold: 高阈值
        line_width: 线条宽度
    
    Returns:
        CannyLineart 实例
    """
    return CannyLineart(low_threshold, high_threshold, line_width)

