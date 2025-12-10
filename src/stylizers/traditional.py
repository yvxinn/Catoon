"""
TraditionalStylizer - 传统风格化器

使用边缘保持平滑 + KMeans 颜色量化实现卡通风格化。
处理流程：原图 → edge-preserving 平滑 → 颜色量化（KMeans）→ toon base
"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from omegaconf import DictConfig

from ..context import StyleCandidate
from .base import BaseStylizer


class TraditionalStylizer(BaseStylizer):
    """传统风格化器（bilateral + KMeans）"""
    
    def __init__(self, style_id: str, cfg: DictConfig):
        """
        初始化传统风格化器
        
        Args:
            style_id: 风格标识符
            cfg: 配置对象
        """
        super().__init__(style_id, cfg)
        self.model_type = "traditional"
        self.model_name = "bilateral_kmeans"
        
        # 获取传统风格化配置
        trad_cfg = cfg.stylizers.traditional
        self.smooth_method = trad_cfg.smooth_method  # bilateral | edge_preserving | mean_shift
        self.default_K = trad_cfg.default_K  # KMeans 颜色数量
        
        # bilateral filter 参数
        self.bilateral_d = trad_cfg.get("bilateral_d", 9)
        self.bilateral_sigma_color = trad_cfg.get("bilateral_sigma_color", 75)
        self.bilateral_sigma_space = trad_cfg.get("bilateral_sigma_space", 75)
    
    def stylize(
        self, 
        image_f32: np.ndarray,
        K: int | None = None,
        smooth_strength: float = 1.0
    ) -> StyleCandidate:
        """
        对图像进行传统风格化
        
        Args:
            image_f32: 输入图像，float32 (H,W,3) [0,1]
            K: 颜色量化的聚类数，默认使用配置值
            smooth_strength: 平滑强度，0~1
        
        Returns:
            StyleCandidate
        """
        K = K or self.default_K
        
        # 转换为 uint8 进行处理
        image_u8 = (image_f32 * 255).astype(np.uint8)
        
        # Step 1: 边缘保持平滑
        smoothed = self._apply_smoothing(image_u8, smooth_strength)
        
        # Step 2: 颜色量化
        quantized = self._quantize_colors(smoothed, K)
        
        # 转回 float32
        result_f32 = quantized.astype(np.float32) / 255.0
        
        # 计算颜色统计
        color_stats = self.compute_color_stats(result_f32)
        
        return StyleCandidate(
            style_id=self.style_id,
            image=result_f32,
            color_stats=color_stats,
            model_type=self.model_type,
            model_name=self.model_name
        )
    
    def _apply_smoothing(
        self, 
        image_u8: np.ndarray, 
        strength: float = 1.0
    ) -> np.ndarray:
        """
        应用边缘保持平滑
        
        Args:
            image_u8: uint8 图像
            strength: 平滑强度 0~1
        
        Returns:
            平滑后的图像
        """
        if strength < 0.01:
            return image_u8
        
        # 根据强度调整参数
        d = max(1, int(self.bilateral_d * strength))
        sigma_color = self.bilateral_sigma_color * strength
        sigma_space = self.bilateral_sigma_space * strength
        
        if self.smooth_method == "bilateral":
            # 双边滤波
            smoothed = cv2.bilateralFilter(
                image_u8, 
                d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        elif self.smooth_method == "edge_preserving":
            # OpenCV 的边缘保持滤波
            smoothed = cv2.edgePreservingFilter(
                image_u8,
                flags=cv2.RECURS_FILTER,  # RECURS_FILTER 或 NORMCONV_FILTER
                sigma_s=60 * strength,
                sigma_r=0.4 * strength
            )
        
        elif self.smooth_method == "mean_shift":
            # 均值漂移滤波（更"海报化"，但较慢）
            sp = int(10 * strength)
            sr = int(30 * strength)
            smoothed = cv2.pyrMeanShiftFiltering(
                image_u8,
                sp=max(1, sp),
                sr=max(1, sr)
            )
        
        else:
            # 默认使用双边滤波
            smoothed = cv2.bilateralFilter(
                image_u8, d=d,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_space
            )
        
        return smoothed
    
    def _quantize_colors(
        self, 
        image_u8: np.ndarray, 
        K: int
    ) -> np.ndarray:
        """
        使用 KMeans 进行颜色量化
        
        Args:
            image_u8: uint8 图像
            K: 聚类数量
        
        Returns:
            量化后的图像
        """
        h, w = image_u8.shape[:2]
        
        # Reshape 为 (N, 3)
        pixels = image_u8.reshape(-1, 3).astype(np.float32)
        
        # 使用 MiniBatchKMeans 更快
        kmeans = MiniBatchKMeans(
            n_clusters=K,
            random_state=42,
            batch_size=1024,
            n_init=3,
            max_iter=100
        )
        
        # 拟合并预测
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        # 用聚类中心替换像素
        quantized = centers[labels].reshape(h, w, 3)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return quantized
    
    def stylize_with_params(
        self,
        image_f32: np.ndarray,
        K: int = 16,
        smooth_strength: float = 0.6,
        edge_strength: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        带参数的风格化，同时返回边缘图
        
        Args:
            image_f32: 输入图像
            K: 颜色数量
            smooth_strength: 平滑强度
            edge_strength: 边缘强度（这里只返回，不叠加）
        
        Returns:
            (stylized_image, edge_map) 元组
        """
        candidate = self.stylize(image_f32, K=K, smooth_strength=smooth_strength)
        
        # 提取边缘
        image_u8 = (image_f32 * 255).astype(np.uint8)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = edges.astype(np.float32) / 255.0
        
        return candidate.image, edges


def create_traditional_stylizer(cfg: DictConfig) -> TraditionalStylizer:
    """
    便捷函数：创建传统风格化器
    
    Args:
        cfg: 配置对象
    
    Returns:
        TraditionalStylizer 实例
    """
    return TraditionalStylizer("Traditional", cfg)

