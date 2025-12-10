"""
XDoG - eXtended Difference of Gaussians 线稿引擎

生成艺术风格的线条，比 Canny 更适合漫画/动画风格。
XDoG 算法产生更平滑、更有艺术感的线条。
"""

import cv2
import numpy as np


class XDoGLineart:
    """XDoG 线稿生成器"""
    
    def __init__(
        self,
        sigma: float = 0.5,
        k: float = 1.6,
        p: float = 19.0,
        epsilon: float = 0.01,
        phi: float = 10.0,
        line_width: int = 1
    ):
        """
        初始化 XDoG 线稿生成器
        
        Args:
            sigma: 第一个高斯核的标准差
            k: 第二个高斯核的倍数 (sigma * k)
            p: 锐化参数
            epsilon: 阈值参数
            phi: 软阈值的陡度
            line_width: 线条宽度
        """
        self.sigma = sigma
        self.k = k
        self.p = p
        self.epsilon = epsilon
        self.phi = phi
        self.line_width = line_width
    
    def extract(self, image_u8: np.ndarray) -> np.ndarray:
        """
        提取 XDoG 线稿
        
        Args:
            image_u8: 输入图像 uint8 (H,W,3) 或 (H,W)
        
        Returns:
            线稿图 float32 (H,W) [0,1]，1 表示线条
        """
        # 转为灰度
        if len(image_u8.shape) == 3:
            gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_u8.copy()
        
        gray = gray.astype(np.float32) / 255.0
        
        # 计算两个高斯模糊
        sigma1 = self.sigma
        sigma2 = self.sigma * self.k
        
        # 高斯核大小（确保为奇数）
        ksize1 = int(np.ceil(sigma1 * 6)) | 1
        ksize2 = int(np.ceil(sigma2 * 6)) | 1
        
        g1 = cv2.GaussianBlur(gray, (ksize1, ksize1), sigma1)
        g2 = cv2.GaussianBlur(gray, (ksize2, ksize2), sigma2)
        
        # Difference of Gaussians with sharpening
        # D(x) = (1 + p) * G_sigma(x) - p * G_(k*sigma)(x)
        dog = (1 + self.p) * g1 - self.p * g2
        
        # 软阈值化
        # T(x) = 1 if D(x) >= epsilon else 1 + tanh(phi * (D(x) - epsilon))
        edges = np.where(
            dog >= self.epsilon,
            1.0,
            1.0 + np.tanh(self.phi * (dog - self.epsilon))
        )
        
        # 归一化到 [0, 1]，反转使线条为白色（高值）
        edges = 1.0 - edges
        edges = np.clip(edges, 0, 1)
        
        # 调整线条宽度
        if self.line_width > 1:
            kernel = np.ones((self.line_width, self.line_width), np.uint8)
            edges_u8 = (edges * 255).astype(np.uint8)
            edges_u8 = cv2.dilate(edges_u8, kernel, iterations=1)
            edges = edges_u8.astype(np.float32) / 255.0
        
        return edges.astype(np.float32)
    
    def overlay(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        将线稿叠加到图像（multiply 模式）
        
        Args:
            image: 输入图像 float32 (H,W,3)
            edges: 线稿图 float32 (H,W) [0,1]，1 表示线条
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
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def extract_with_params(
        self,
        image_u8: np.ndarray,
        sigma: float | None = None,
        k: float | None = None,
        p: float | None = None,
        epsilon: float | None = None,
        phi: float | None = None
    ) -> np.ndarray:
        """
        使用自定义参数提取线稿
        
        Args:
            image_u8: 输入图像
            sigma, k, p, epsilon, phi: XDoG 参数
        
        Returns:
            线稿图
        """
        # 保存原始参数
        orig = (self.sigma, self.k, self.p, self.epsilon, self.phi)
        
        # 临时设置
        if sigma is not None:
            self.sigma = sigma
        if k is not None:
            self.k = k
        if p is not None:
            self.p = p
        if epsilon is not None:
            self.epsilon = epsilon
        if phi is not None:
            self.phi = phi
        
        # 提取
        result = self.extract(image_u8)
        
        # 恢复
        self.sigma, self.k, self.p, self.epsilon, self.phi = orig
        
        return result


def create_xdog_lineart(
    sigma: float = 0.5,
    k: float = 1.6,
    p: float = 19.0,
    epsilon: float = 0.01,
    phi: float = 10.0,
    line_width: int = 1
) -> XDoGLineart:
    """
    便捷函数：创建 XDoG 线稿生成器
    
    Args:
        sigma, k, p, epsilon, phi: XDoG 参数
        line_width: 线条宽度
    
    Returns:
        XDoGLineart 实例
    """
    return XDoGLineart(sigma, k, p, epsilon, phi, line_width)

