"""
TraditionalStylizer - 传统风格化器

使用边缘保持平滑 + KMeans 颜色量化实现卡通风格化。
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
        super().__init__(style_id, cfg)
        self.model_type = "traditional"
        self.model_name = "bilateral_kmeans"
        
        trad_cfg = cfg.stylizers.traditional
        self.smooth_method = trad_cfg.smooth_method
        self.default_K = trad_cfg.default_K
        self.bilateral_d = trad_cfg.get("bilateral_d", 9)
        self.bilateral_sigma_color = trad_cfg.get("bilateral_sigma_color", 75)
        self.bilateral_sigma_space = trad_cfg.get("bilateral_sigma_space", 75)
    
    def stylize(
        self, 
        image_f32: np.ndarray,
        K: int | None = None,
        smooth_strength: float = 1.0,
        smooth_method: str | None = None
    ) -> StyleCandidate:
        """对图像进行传统风格化"""
        K = K or self.default_K
        
        orig_method = self.smooth_method
        if smooth_method is not None:
            self.smooth_method = smooth_method
        
        image_u8 = (image_f32 * 255).astype(np.uint8)
        smoothed = self._apply_smoothing(image_u8, smooth_strength)
        quantized = self._quantize_colors(smoothed, K)
        result_f32 = quantized.astype(np.float32) / 255.0
        color_stats = self.compute_color_stats(result_f32)
        
        self.smooth_method = orig_method
        
        return StyleCandidate(
            style_id=self.style_id,
            image=result_f32,
            color_stats=color_stats,
            model_type=self.model_type,
            model_name=self.model_name
        )
    
    def _apply_smoothing(self, image_u8: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """应用边缘保持平滑"""
        if strength < 0.01:
            return image_u8
        
        d = max(1, int(self.bilateral_d * strength))
        sigma_color = self.bilateral_sigma_color * strength
        sigma_space = self.bilateral_sigma_space * strength
        
        if self.smooth_method == "bilateral":
            smoothed = cv2.bilateralFilter(image_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        elif self.smooth_method == "edge_preserving":
            smoothed = cv2.edgePreservingFilter(image_u8, flags=cv2.RECURS_FILTER, sigma_s=60 * strength, sigma_r=0.4 * strength)
        elif self.smooth_method == "mean_shift":
            sp = int(10 * strength)
            sr = int(30 * strength)
            smoothed = cv2.pyrMeanShiftFiltering(image_u8, sp=max(1, sp), sr=max(1, sr))
        else:
            smoothed = cv2.bilateralFilter(image_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        
        return smoothed
    
    def _quantize_colors(self, image_u8: np.ndarray, K: int) -> np.ndarray:
        """使用 KMeans 进行颜色量化"""
        h, w = image_u8.shape[:2]
        pixels = image_u8.reshape(-1, 3).astype(np.float32)
        
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=1024, n_init=3, max_iter=100)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        quantized = centers[labels].reshape(h, w, 3)
        return np.clip(quantized, 0, 255).astype(np.uint8)
    
    def stylize_with_params(
        self,
        image_f32: np.ndarray,
        K: int = 16,
        smooth_strength: float = 0.6,
        edge_strength: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """带参数的风格化，同时返回边缘图"""
        candidate = self.stylize(image_f32, K=K, smooth_strength=smooth_strength)
        
        image_u8 = (image_f32 * 255).astype(np.uint8)
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = edges.astype(np.float32) / 255.0
        
        return candidate.image, edges


def create_traditional_stylizer(cfg: DictConfig) -> TraditionalStylizer:
    """便捷函数：创建传统风格化器"""
    return TraditionalStylizer("Traditional", cfg)

