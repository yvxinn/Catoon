"""
BaseStylizer - 风格化器基类

定义风格化器的通用接口。
"""

from abc import ABC, abstractmethod
import numpy as np
from omegaconf import DictConfig

from ..context import StyleCandidate, ColorStats


class BaseStylizer(ABC):
    """风格化器基类"""
    
    def __init__(self, style_id: str, cfg: DictConfig):
        """
        初始化风格化器
        
        Args:
            style_id: 风格标识符
            cfg: 配置对象
        """
        self.style_id = style_id
        self.cfg = cfg
        self.model_type = "base"
        self.model_name = ""
    
    @abstractmethod
    def stylize(self, image_f32: np.ndarray) -> StyleCandidate:
        """
        对图像进行风格化
        
        Args:
            image_f32: 输入图像，float32 (H,W,3) [0,1]
        
        Returns:
            StyleCandidate 包含风格化后的图像和颜色统计
        """
        pass
    
    @staticmethod
    def compute_color_stats(image_f32: np.ndarray) -> ColorStats:
        """
        计算图像的颜色统计信息
        
        Args:
            image_f32: float32 图像 [0,1]
        
        Returns:
            ColorStats 对象
        """
        import cv2
        
        # 转换为 uint8 再转 Lab
        image_u8 = (image_f32 * 255).astype(np.uint8)
        lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 计算 Lab 空间的均值和标准差
        lab_mean = lab.mean(axis=(0, 1))
        lab_std = lab.std(axis=(0, 1))
        
        # 计算直方图（可选）
        histogram = np.zeros((3, 256), dtype=np.float32)
        for i in range(3):
            histogram[i], _ = np.histogram(
                image_u8[:, :, i].flatten(), 
                bins=256, 
                range=(0, 256)
            )
            histogram[i] /= histogram[i].sum() + 1e-8  # 归一化
        
        return ColorStats(
            lab_mean=lab_mean,
            lab_std=lab_std,
            histogram=histogram
        )


def init_stylizers(cfg: DictConfig) -> dict[str, BaseStylizer]:
    """
    初始化所有配置的风格化器
    
    Args:
        cfg: 配置对象
    
    Returns:
        {style_id: stylizer} 字典
    """
    from .traditional import TraditionalStylizer
    from .animegan import AnimeGANStylizer
    
    stylizers = {}
    
    # 初始化传统风格化器
    trad_stylizer = TraditionalStylizer("Traditional", cfg)
    stylizers["Traditional"] = trad_stylizer
    
    # 初始化 AnimeGAN 风格化器（Phase 2）
    gan_configs = cfg.stylizers.get("gan", [])
    for gan_cfg in gan_configs:
        try:
            style_name = gan_cfg.name
            if style_name in AnimeGANStylizer.AVAILABLE_STYLES:
                stylizer = AnimeGANStylizer(style_name, cfg)
                stylizers[style_name] = stylizer
                print(f"[Stylizers] Initialized {style_name} stylizer")
        except Exception as e:
            print(f"[Stylizers] Failed to init {gan_cfg.name}: {e}")
    
    return stylizers

