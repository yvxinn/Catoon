"""
Preprocess 模块 - 图像预处理

职责：
- 输入图像统一到标准格式与尺度
- 维护 Context（全局元信息、缓存、缩放回原图）
"""

from .preprocessor import Preprocessor

__all__ = ["Preprocessor"]

