"""
Routing 模块 - 语义路由

职责：
- 把"语义 mask + 用户配置"转换为每个区域的风格选择与参数
- 人脸保护策略对路由进行 override
"""

from .router import SemanticRouter

__all__ = ["SemanticRouter"]

