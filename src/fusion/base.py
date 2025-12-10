"""
FusionModule - 融合模块入口

根据配置选择不同的融合方法。
"""

import numpy as np
from omegaconf import DictConfig

from ..context import RoutingPlan, SegmentationOutput, StyleCandidate


class FusionModule:
    """融合模块"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化融合模块
        
        Args:
            cfg: 配置对象
        """
        self.cfg = cfg.fusion
        self.default_method = self.cfg.default_method
        self.blur_kernel = self.cfg.soft_mask_blur_kernel
        self.pyramid_levels = self.cfg.get("pyramid_levels", 6)
        
        # 延迟加载具体融合器
        self._soft_mask_fuser = None
        self._pyramid_fuser = None
    
    @property
    def soft_mask_fuser(self):
        """懒加载 soft mask 融合器"""
        if self._soft_mask_fuser is None:
            from .soft_mask import SoftMaskFusion
            self._soft_mask_fuser = SoftMaskFusion(self.blur_kernel)
        return self._soft_mask_fuser
    
    @property
    def pyramid_fuser(self):
        """懒加载 Laplacian pyramid 融合器"""
        if self._pyramid_fuser is None:
            from .pyramid import LaplacianPyramidFusion
            self._pyramid_fuser = LaplacianPyramidFusion(
                levels=self.pyramid_levels,
                blur_kernel=self.blur_kernel
            )
        return self._pyramid_fuser
    
    def fuse(
        self,
        candidates: dict[str, StyleCandidate],
        routing: RoutingPlan,
        seg_out: SegmentationOutput,
        method: str | None = None
    ) -> np.ndarray:
        """
        融合候选图像
        
        Args:
            candidates: {style_id: StyleCandidate} 候选字典
            routing: 路由计划
            seg_out: 分割输出
            method: 融合方法，默认使用配置
        
        Returns:
            融合后的图像 float32 (H,W,3) [0,1]
        """
        method = method or self.default_method
        
        if method == "soft_mask":
            return self.soft_mask_fuser.fuse(
                candidates, routing, seg_out
            )
        
        elif method == "laplacian_pyramid":
            return self.pyramid_fuser.fuse(
                candidates, routing, seg_out
            )
        
        elif method == "poisson":
            # TODO: Phase 3 实现
            # 暂时 fallback 到 pyramid
            return self.pyramid_fuser.fuse(
                candidates, routing, seg_out
            )
        
        else:
            # 默认使用 soft_mask
            return self.soft_mask_fuser.fuse(
                candidates, routing, seg_out
            )

