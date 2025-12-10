"""
AnimeGANStylizer - AnimeGAN 风格化器

使用 AnimeGANv2 生成动漫风格图像。
支持多种预训练风格：Hayao (宫崎骏)、Shinkai (新海诚)、Paprika (今敏)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..context import StyleCandidate
from .base import BaseStylizer


class AnimeGANv2Generator(nn.Module):
    """AnimeGANv2 生成器网络"""
    
    def __init__(self):
        super().__init__()
        # 简化的 AnimeGAN 架构
        # 使用 torch.hub 加载预训练模型
        self.model = None
    
    def forward(self, x):
        if self.model is not None:
            return self.model(x)
        return x


class AnimeGANStylizer(BaseStylizer):
    """AnimeGAN 风格化器"""
    
    # 可用的风格和对应的 Hub 模型名
    AVAILABLE_STYLES = {
        "Hayao": "Hayao",      # 宫崎骏风格
        "Shinkai": "Shinkai",  # 新海诚风格
        "Paprika": "Paprika",  # 今敏风格
    }
    
    def __init__(self, style_id: str, cfg: DictConfig):
        """
        初始化 AnimeGAN 风格化器
        
        Args:
            style_id: 风格标识符 (Hayao, Shinkai, Paprika)
            cfg: 配置对象
        """
        super().__init__(style_id, cfg)
        self.model_type = "gan"
        self.model_name = f"AnimeGANv2_{style_id}"
        
        if style_id not in self.AVAILABLE_STYLES:
            raise ValueError(f"Unknown style: {style_id}. Available: {list(self.AVAILABLE_STYLES.keys())}")
        
        self.style_name = self.AVAILABLE_STYLES[style_id]
        
        # 解析设备
        global_cfg = getattr(cfg, 'global')
        device_str = global_cfg.device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        # 延迟加载模型
        self._model = None
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """加载 AnimeGAN 模型"""
        print(f"[AnimeGAN] Loading model: {self.style_name}")
        
        try:
            # 尝试从 torch.hub 加载
            self._model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator",
                pretrained=self.style_name,
                device=self.device,
            )
            self._model.eval()
            print(f"[AnimeGAN] Model loaded on {self.device}")
        except Exception as e:
            print(f"[AnimeGAN] Failed to load from hub: {e}")
            print("[AnimeGAN] Falling back to Traditional stylizer")
            self._model = None
    
    def stylize(self, image_f32: np.ndarray) -> StyleCandidate:
        """
        对图像进行 AnimeGAN 风格化
        
        Args:
            image_f32: 输入图像，float32 (H,W,3) [0,1]
        
        Returns:
            StyleCandidate
        """
        if self._model is None:
            # 模型加载失败，使用简单的色彩调整作为 fallback
            result = self._fallback_stylize(image_f32)
        else:
            result = self._gan_stylize(image_f32)
        
        # 计算颜色统计
        color_stats = self.compute_color_stats(result)
        
        return StyleCandidate(
            style_id=self.style_id,
            image=result,
            color_stats=color_stats,
            model_type=self.model_type,
            model_name=self.model_name
        )
    
    def _gan_stylize(self, image_f32: np.ndarray) -> np.ndarray:
        """使用 GAN 进行风格化"""
        h, w = image_f32.shape[:2]
        
        # 转换为 tensor
        # AnimeGAN 期望 RGB [0, 1] -> [-1, 1]
        image_tensor = torch.from_numpy(image_f32).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor * 2 - 1  # [0,1] -> [-1,1]
        image_tensor = image_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # 转换回 numpy
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = (output + 1) / 2  # [-1,1] -> [0,1]
        output = np.clip(output, 0, 1).astype(np.float32)
        
        # 如果尺寸改变，resize 回原尺寸
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return output
    
    def _fallback_stylize(self, image_f32: np.ndarray) -> np.ndarray:
        """
        Fallback 风格化（当 GAN 不可用时）
        使用简单的色彩调整模拟不同风格
        """
        result = image_f32.copy()
        
        if self.style_id == "Shinkai":
            # 新海诚风格：增强蓝色、提高饱和度和对比度
            result = self._adjust_colors(result, 
                                         saturation=1.3, 
                                         contrast=1.1,
                                         blue_boost=1.15)
        elif self.style_id == "Hayao":
            # 宫崎骏风格：温暖色调、柔和
            result = self._adjust_colors(result,
                                         saturation=1.1,
                                         contrast=1.05,
                                         warmth=0.1)
        elif self.style_id == "Paprika":
            # 今敏风格：高饱和、鲜艳
            result = self._adjust_colors(result,
                                         saturation=1.4,
                                         contrast=1.15)
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def _adjust_colors(
        self, 
        image: np.ndarray,
        saturation: float = 1.0,
        contrast: float = 1.0,
        blue_boost: float = 1.0,
        warmth: float = 0.0
    ) -> np.ndarray:
        """调整图像颜色"""
        result = image.copy()
        
        # 转换到 HSV 调整饱和度
        if abs(saturation - 1.0) > 0.01:
            result_u8 = (result * 255).astype(np.uint8)
            hsv = cv2.cvtColor(result_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            result_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = result_u8.astype(np.float32) / 255.0
        
        # 对比度
        if abs(contrast - 1.0) > 0.01:
            mean = result.mean()
            result = (result - mean) * contrast + mean
        
        # 蓝色增强
        if abs(blue_boost - 1.0) > 0.01:
            result[:, :, 2] = result[:, :, 2] * blue_boost
        
        # 色温（正值偏暖，负值偏冷）
        if abs(warmth) > 0.01:
            result[:, :, 0] = result[:, :, 0] + warmth * 0.1  # R
            result[:, :, 2] = result[:, :, 2] - warmth * 0.05  # B
        
        return np.clip(result, 0, 1)
    
    @classmethod
    def get_available_styles(cls) -> list[str]:
        """获取可用的风格列表"""
        return list(cls.AVAILABLE_STYLES.keys())


def create_animegan_stylizer(style_id: str, cfg: DictConfig) -> AnimeGANStylizer:
    """
    便捷函数：创建 AnimeGAN 风格化器
    
    Args:
        style_id: 风格 ID
        cfg: 配置对象
    
    Returns:
        AnimeGANStylizer 实例
    """
    return AnimeGANStylizer(style_id, cfg)

