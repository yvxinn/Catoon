"""
Stylizers 模块 - 风格化生成器

职责：
- 对整图生成多个风格候选
- 包括 GAN 风格化和传统风格化
- 缓存候选避免重复推理
"""

from .traditional import TraditionalStylizer
from .animegan import AnimeGANStylizer
from .diffusion import DiffusionStylizer
from .base import BaseStylizer, init_stylizers

__all__ = [
    "TraditionalStylizer",
    "AnimeGANStylizer",
    "DiffusionStylizer",
    "BaseStylizer",
    "init_stylizers"
]

