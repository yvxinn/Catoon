"""
LineartEngine - 线稿引擎入口

根据配置选择不同的线稿生成方法。
支持 Canny (稳定) 和 XDoG (艺术风格)。
"""

import numpy as np
from omegaconf import DictConfig


class LineartEngine:
    """线稿引擎"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化线稿引擎
        
        Args:
            cfg: 配置对象
        """
        self.cfg = cfg.lineart
        self.default_engine = self.cfg.engine
        self.default_strength = self.cfg.default_strength
        self.canny_low = self.cfg.canny_low
        self.canny_high = self.cfg.canny_high
        self.line_width = self.cfg.line_width
        
        # XDoG 参数（Phase 3）
        self.xdog_sigma = self.cfg.get("xdog_sigma", 0.5)
        self.xdog_k = self.cfg.get("xdog_k", 1.6)
        self.xdog_p = self.cfg.get("xdog_p", 19.0)
        
        # 延迟加载具体引擎
        self._canny_engine = None
        self._xdog_engine = None
        self._guided_filter = None
    
    @property
    def canny_engine(self):
        """懒加载 Canny 引擎"""
        if self._canny_engine is None:
            from .canny import CannyLineart
            self._canny_engine = CannyLineart(
                low_threshold=self.canny_low,
                high_threshold=self.canny_high,
                line_width=self.line_width
            )
        return self._canny_engine
    
    @property
    def xdog_engine(self):
        """懒加载 XDoG 引擎"""
        if self._xdog_engine is None:
            from .xdog import XDoGLineart
            self._xdog_engine = XDoGLineart(
                sigma=self.xdog_sigma,
                k=self.xdog_k,
                p=self.xdog_p,
                line_width=self.line_width
            )
        return self._xdog_engine
    
    @property
    def guided_filter(self):
        """懒加载 Guided Filter 增强器"""
        if self._guided_filter is None:
            from .guided_filter import GuidedFilterEnhancer
            self._guided_filter = GuidedFilterEnhancer(
                radius=8,
                eps=0.02,
                detail_strength=0.5
            )
        return self._guided_filter
    
    def extract(
        self,
        image_u8: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """
        提取线稿
        
        Args:
            image_u8: 输入图像 uint8
            params: 参数字典
        
        Returns:
            边缘图 float32 (H,W) [0,1]
        """
        engine = params.get("line_engine", self.default_engine)
        
        if engine == "canny":
            return self.canny_engine.extract(image_u8)
        
        elif engine == "xdog":
            return self.xdog_engine.extract(image_u8)
        
        else:
            return self.canny_engine.extract(image_u8)
    
    def overlay(
        self,
        image: np.ndarray,
        edges: np.ndarray,
        strength: float,
        params: dict
    ) -> np.ndarray:
        """
        将线稿叠加到图像
        
        Args:
            image: 输入图像 float32
            edges: 边缘图 float32
            strength: 线稿强度
            params: 参数字典
        
        Returns:
            叠加后的图像 float32
        """
        engine = params.get("line_engine", self.default_engine)
        
        if engine == "canny":
            return self.canny_engine.overlay(image, edges, strength)
        elif engine == "xdog":
            return self.xdog_engine.overlay(image, edges, strength)
        else:
            return self.canny_engine.overlay(image, edges, strength)
    
    def enhance_detail(
        self,
        image: np.ndarray,
        guide: np.ndarray | None = None,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        使用 Guided Filter 增强细节
        
        Args:
            image: 输入图像 float32
            guide: 引导图像（原图），可选
            strength: 增强强度
        
        Returns:
            增强后的图像 float32
        """
        if strength < 0.01:
            return image
        
        self.guided_filter.detail_strength = strength
        return self.guided_filter.enhance(image, guide)

