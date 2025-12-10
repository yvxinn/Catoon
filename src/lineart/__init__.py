"""
Lineart 模块 - 线稿生成

职责：
- 输出可控线稿叠加效果
- 支持 Canny 和 XDoG 引擎
- Guided Filter 细节注入（含降级方案）
"""

from .canny import CannyLineart
from .xdog import XDoGLineart
from .guided_filter import GuidedFilterEnhancer
from .engine import LineartEngine

__all__ = [
    "CannyLineart",
    "XDoGLineart",
    "GuidedFilterEnhancer",
    "LineartEngine"
]

