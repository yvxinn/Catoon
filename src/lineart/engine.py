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
            params: 参数字典 (line_engine, line_width, canny_low/high, xdog_sigma/k/p)
        
        Returns:
            边缘图 float32 (H,W) [0,1]
        """
        engine = params.get("line_engine", self.default_engine)
        
        # 动态更新参数
        line_width = params.get("line_width", self.line_width)
        
        if engine == "canny":
            # 动态更新 Canny 参数
            canny_low = params.get("canny_low", self.canny_low)
            canny_high = params.get("canny_high", self.canny_high)
            self.canny_engine.low_threshold = canny_low
            self.canny_engine.high_threshold = canny_high
            self.canny_engine.line_width = line_width
            return self.canny_engine.extract(image_u8)
        
        elif engine == "xdog":
            # 动态更新 XDoG 参数
            xdog_sigma = params.get("xdog_sigma", self.xdog_sigma)
            xdog_k = params.get("xdog_k", self.xdog_k)
            xdog_p = params.get("xdog_p", self.xdog_p)
            self.xdog_engine.sigma = xdog_sigma
            self.xdog_engine.k = xdog_k
            self.xdog_engine.p = xdog_p
            self.xdog_engine.line_width = line_width
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
    
    def extract_from_stylized(
        self,
        stylized_image: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """
        从风格化后的图像提取线稿（用于后期线稿增强）
        
        Args:
            stylized_image: 风格化后的图像 float32 [0,1]
            params: 参数字典
        
        Returns:
            边缘图 float32 (H,W) [0,1]
        """
        # 转换为 uint8
        image_u8 = (np.clip(stylized_image, 0, 1) * 255).astype(np.uint8)
        return self.extract(image_u8, params)
    
    def overlay_with_semantic_routing(
        self,
        image: np.ndarray,
        semantic_masks: dict,
        region_configs: dict,
        params: dict
    ) -> np.ndarray:
        """
        按语义区域分别应用线稿叠加（每个区域使用独立的引擎和参数）
        
        Args:
            image: 风格化后的图像 float32 [0,1]
            semantic_masks: {bucket_name: mask} 语义掩码
            region_configs: {bucket_name: RegionConfig} 区域配置
            params: 全局参数字典（作为默认值）
        
        Returns:
            叠加线稿后的图像 float32
        """
        import cv2
        
        result = image.copy()
        h, w = image.shape[:2]
        image_u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # 按区域应用不同的线稿参数
        for bucket_name, mask in semantic_masks.items():
            config = region_configs.get(bucket_name)
            if config is None:
                continue
            
            # 获取区域级线稿强度
            lineart_strength = getattr(config, "lineart_strength", 0.5)
            if lineart_strength < 0.01:
                continue
            
            # 调整 mask 尺寸以匹配图像
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 获取区域级线稿引擎参数
            region_engine = getattr(config, "line_engine", params.get("line_engine", "canny"))
            region_line_width = getattr(config, "line_width", params.get("line_width", 1.0))
            
            # 构建区域级参数
            region_params = {
                "line_engine": region_engine,
                "line_width": region_line_width,
                "canny_low": getattr(config, "canny_low", params.get("canny_low", 100)),
                "canny_high": getattr(config, "canny_high", params.get("canny_high", 200)),
                "xdog_sigma": getattr(config, "xdog_sigma", params.get("xdog_sigma", 0.5)),
                "xdog_k": getattr(config, "xdog_k", params.get("xdog_k", 1.6)),
                "xdog_p": getattr(config, "xdog_p", params.get("xdog_p", 19.0)),
            }
            
            # 使用区域参数提取边缘
            edges = self.extract(image_u8, region_params)
            
            # 选择叠加方法
            if region_engine == "xdog":
                overlay_fn = self.xdog_engine.overlay
            else:
                overlay_fn = self.canny_engine.overlay
            
            # 对该区域应用线稿
            region_with_edges = overlay_fn(image, edges, lineart_strength)
            
            # 使用 mask 混合
            mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
            result = result * (1 - mask_3d) + region_with_edges * mask_3d
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def enhance_detail_with_semantic_routing(
        self,
        image: np.ndarray,
        guide: np.ndarray,
        semantic_masks: dict,
        region_configs: dict
    ) -> np.ndarray:
        """
        按语义区域分别应用细节增强
        
        Args:
            image: 风格化后的图像 float32 [0,1]
            guide: 引导图像（原图）float32 [0,1]
            semantic_masks: {bucket_name: mask} 语义掩码
            region_configs: {bucket_name: RegionConfig} 区域配置
        
        Returns:
            增强后的图像 float32
        """
        import cv2
        
        result = image.copy()
        h, w = image.shape[:2]
        
        for bucket_name, mask in semantic_masks.items():
            config = region_configs.get(bucket_name)
            if config is None:
                continue
            
            # 获取区域级细节增强强度
            detail_enhance = getattr(config, "detail_enhance", 0.0)
            if detail_enhance < 0.01:
                continue
            
            # 调整 mask 尺寸以匹配图像
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 对该区域应用细节增强
            enhanced = self.enhance_detail(image, guide, detail_enhance)
            
            # 使用 mask 混合
            mask_3d = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
            result = result * (1 - mask_3d) + enhanced * mask_3d
        
        return np.clip(result, 0, 1).astype(np.float32)

