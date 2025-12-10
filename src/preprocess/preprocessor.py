"""
Preprocessor - 图像预处理器

核心功能：
- resize_max_side: 最长边限制为 max_image_size（默认 1024）
- normalize: 输出 float32 RGB，范围 [0,1]
- 创建 Context 管理全局状态
"""

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from ..context import Context


class Preprocessor:
    """图像预处理器"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化预处理器
        
        Args:
            cfg: 配置对象，需包含 global.max_image_size 和 global.device
        """
        # 使用 getattr 访问 'global' 因为它是 Python 保留字
        global_cfg = getattr(cfg, 'global')
        self.max_image_size = global_cfg.max_image_size
        self.device = self._resolve_device(global_cfg.device)
    
    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        解析设备配置
        
        Args:
            device: "auto" | "cuda" | "cpu"
        
        Returns:
            实际使用的设备名
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def process(self, image_u8: np.ndarray) -> Context:
        """
        预处理输入图像
        
        Args:
            image_u8: 输入图像，uint8 (H,W,3) RGB 或 BGR
        
        Returns:
            Context 对象，包含预处理后的图像和元信息
        
        Raises:
            ValueError: 如果输入图像格式不正确
        """
        # 验证输入
        if image_u8 is None:
            raise ValueError("输入图像不能为空")
        if image_u8.ndim != 3 or image_u8.shape[2] != 3:
            raise ValueError(f"输入图像必须是 (H,W,3) 格式，当前: {image_u8.shape}")
        if image_u8.dtype != np.uint8:
            # 尝试转换
            image_u8 = np.clip(image_u8, 0, 255).astype(np.uint8)
        
        # 保存原始尺寸
        orig_h, orig_w = image_u8.shape[:2]
        orig_size = (orig_h, orig_w)
        
        # Resize（保持长宽比，最长边不超过 max_image_size）
        image_resized, proc_size, scale = self._resize_max_side(
            image_u8, self.max_image_size
        )
        
        # Normalize to float32 [0, 1]
        image_f32 = self._normalize(image_resized)
        
        # 创建 Context
        ctx = Context(
            image_u8=image_resized,
            image_f32=image_f32,
            orig_size=orig_size,
            proc_size=proc_size,
            scale=scale,
            device=self.device
        )
        
        return ctx
    
    def postprocess(self, image: np.ndarray, ctx: Context) -> np.ndarray:
        """
        后处理：将处理后的图像恢复到原始尺寸
        
        Args:
            image: 处理后的图像，float32 (H,W,3) [0,1] 或 uint8
            ctx: Context 对象
        
        Returns:
            恢复到原始尺寸的 uint8 图像
        """
        # 转换为 uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            image_u8 = image
        
        # 恢复到原始尺寸
        orig_h, orig_w = ctx.orig_size
        if image_u8.shape[:2] != (orig_h, orig_w):
            image_u8 = cv2.resize(
                image_u8, 
                (orig_w, orig_h),  # cv2.resize 使用 (width, height)
                interpolation=cv2.INTER_LANCZOS4
            )
        
        return image_u8
    
    def _resize_max_side(
        self, 
        image: np.ndarray, 
        max_size: int
    ) -> tuple[np.ndarray, tuple[int, int], float]:
        """
        按最长边缩放图像
        
        Args:
            image: 输入图像
            max_size: 最长边最大值
        
        Returns:
            (resized_image, (new_h, new_w), scale)
        """
        h, w = image.shape[:2]
        max_side = max(h, w)
        
        if max_side <= max_size:
            # 不需要缩放
            return image, (h, w), 1.0
        
        scale = max_size / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(
            image, 
            (new_w, new_h),  # cv2.resize 使用 (width, height)
            interpolation=cv2.INTER_AREA  # 缩小时使用 INTER_AREA 效果更好
        )
        
        return resized, (new_h, new_w), scale
    
    @staticmethod
    def _normalize(image_u8: np.ndarray) -> np.ndarray:
        """
        归一化图像到 [0, 1]
        
        Args:
            image_u8: uint8 图像
        
        Returns:
            float32 图像，范围 [0, 1]
        """
        return image_u8.astype(np.float32) / 255.0


def create_preprocessor(cfg: DictConfig) -> Preprocessor:
    """
    便捷函数：创建预处理器
    
    Args:
        cfg: 配置对象
    
    Returns:
        Preprocessor 实例
    """
    return Preprocessor(cfg)

