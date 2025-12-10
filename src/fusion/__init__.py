"""
Fusion 模块 - 区域融合

职责：
- 将不同区域对应的候选风格图融合为一张 base 图
- 解决 halo/接缝伪影
"""

from .soft_mask import SoftMaskFusion
from .pyramid import LaplacianPyramidFusion
from .base import FusionModule

__all__ = ["SoftMaskFusion", "LaplacianPyramidFusion", "FusionModule"]

