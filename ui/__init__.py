"""
UI 模块 - Gradio 交互界面

模块结构：
- state.py: 会话状态管理 (ProcessingState)
- config.py: 参数数据类和常量
- components.py: UI 组件工厂函数
- theme.py: CSS 和主题定义
- layout.py: UI 布局定义
- logic.py: 核心业务逻辑
- gradio_app.py: 应用入口

使用方式：
    from ui import main
    main()  # 启动 Gradio 应用
"""

from .state import ProcessingState, compute_image_hash
from .config import (
    SEMANTIC_BUCKETS,
    STYLE_CHOICES,
    SEMANTIC_COLORS,
    RegionParams,
    GlobalParams,
    REGION_DEFAULTS,
    build_ui_params,
)
from .gradio_app import main

__all__ = [
    "ProcessingState",
    "compute_image_hash",
    "SEMANTIC_BUCKETS",
    "STYLE_CHOICES",
    "SEMANTIC_COLORS",
    "RegionParams",
    "GlobalParams",
    "REGION_DEFAULTS",
    "build_ui_params",
    "main",
]

