"""
CartoonPipeline - 主处理流水线

语义感知可控卡通化框架的核心入口。
"""

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf, DictConfig

from .context import Context, UIParams


class CartoonPipeline:
    """语义感知可控卡通化主 Pipeline"""
    
    def __init__(self, config_path: str | Path | None = None):
        """
        初始化 Pipeline
        
        Args:
            config_path: 配置文件路径，默认使用 config/default.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        self.cfg: DictConfig = OmegaConf.load(config_path)
        
        # 初始化各模块（延迟加载）
        self._preprocessor = None
        self._segmenter = None
        self._face_detector = None
        self._stylizers = None
        self._router = None
        self._fuser = None
        self._harmonizer = None
        self._lineart = None
        self._depth_enhancer = None
    
    # ==================== 模块懒加载 ====================
    
    @property
    def preprocessor(self):
        """预处理模块（懒加载）"""
        if self._preprocessor is None:
            from .preprocess import Preprocessor
            self._preprocessor = Preprocessor(self.cfg)
        return self._preprocessor
    
    @property
    def segmenter(self):
        """语义分割模块（懒加载）"""
        if self._segmenter is None:
            from .segmentation import SegFormerSegmenter
            self._segmenter = SegFormerSegmenter(self.cfg)
        return self._segmenter
    
    @property
    def face_detector(self):
        """人脸检测模块（懒加载）"""
        if self._face_detector is None and self.cfg.segmentation.use_face_detector:
            from .segmentation import FaceDetector
            self._face_detector = FaceDetector(self.cfg)
        return self._face_detector
    
    @property
    def stylizers(self):
        """风格化器集合（懒加载）"""
        if self._stylizers is None:
            from .stylizers import init_stylizers
            self._stylizers = init_stylizers(self.cfg)
        return self._stylizers
    
    @property
    def router(self):
        """语义路由模块（懒加载）"""
        if self._router is None:
            from .routing import SemanticRouter
            self._router = SemanticRouter(self.cfg)
        return self._router
    
    @property
    def fuser(self):
        """区域融合模块（懒加载）"""
        if self._fuser is None:
            from .fusion import FusionModule
            self._fuser = FusionModule(self.cfg)
        return self._fuser
    
    @property
    def harmonizer(self):
        """全局协调模块（懒加载）"""
        if self._harmonizer is None:
            from .harmonization import Harmonizer
            self._harmonizer = Harmonizer(self.cfg)
        return self._harmonizer
    
    @property
    def lineart(self):
        """线稿模块（懒加载）"""
        if self._lineart is None:
            from .lineart import LineartEngine
            self._lineart = LineartEngine(self.cfg)
        return self._lineart
    
    @property
    def depth_enhancer(self):
        """深度增强模块（懒加载，可选）"""
        if self._depth_enhancer is None and self.cfg.depth.enabled:
            from .depth import DepthEnhancer
            self._depth_enhancer = DepthEnhancer(self.cfg)
        return self._depth_enhancer
    
    # ==================== 主处理流程 ====================
    
    def process(
        self,
        image_u8: np.ndarray,
        ui_params: dict[str, Any] | UIParams | None = None
    ) -> np.ndarray:
        """
        处理单张图像
        
        Args:
            image_u8: 输入图像，uint8 (H,W,3) RGB
            ui_params: UI 参数覆盖，可以是 dict 或 UIParams
        
        Returns:
            处理后的图像，uint8 (H,W,3) RGB
        """
        # 统一 ui_params 格式
        if ui_params is None:
            ui_params = {}
        elif isinstance(ui_params, UIParams):
            ui_params = vars(ui_params)
        
        # A. 预处理
        ctx = self.preprocessor.process(image_u8)
        
        # B. 语义分析
        seg_out = self.segmenter.predict(ctx.image_f32)
        face_mask = None
        if self.face_detector:
            face_mask = self.face_detector.detect(ctx.image_u8)
        
        # C. 风格候选生成（带缓存）
        candidates = self._get_or_build_candidates(ctx)
        
        # D. 语义路由
        routing = self.router.route(
            semantic_masks=seg_out.semantic_masks,
            face_mask=face_mask,
            ui_overrides=ui_params
        )
        
        # E. 区域融合
        fused = self.fuser.fuse(
            candidates=candidates,
            routing=routing,
            seg_out=seg_out,
            method=ui_params.get("fusion_method", self.cfg.fusion.default_method)
        )
        
        # F. 全局协调
        if ui_params.get("harmonization_enabled", self.cfg.harmonization.enabled):
            ref = self.harmonizer.pick_reference(
                candidates, seg_out, ui_params, self.cfg.harmonization
            )
            fused = self.harmonizer.match_and_adjust(fused, ref, ui_params)
        
        # G. 线稿叠加
        edge_strength = ui_params.get("edge_strength", self.cfg.lineart.default_strength)
        if edge_strength > 1e-3:
            edges = self.lineart.extract(ctx.image_u8, ui_params)
            fused = self.lineart.overlay(fused, edges, edge_strength, ui_params)
        
        # H. 深度增强（可选）
        if self.depth_enhancer and ui_params.get("depth_fog_enabled", False):
            depth_map = self.depth_enhancer.estimate(ctx.image_u8)
            fused = self.depth_enhancer.apply_fog(fused, depth_map, ui_params)
        
        # 后处理（恢复原始尺寸）
        out_u8 = self.preprocessor.postprocess(fused, ctx)
        return out_u8
    
    def _get_or_build_candidates(self, ctx: Context) -> dict:
        """获取或构建风格候选（带缓存）"""
        cache_key = ctx.make_cache_key("candidates")
        cached = ctx.get_cache(cache_key)
        if cached is not None:
            return cached
        
        # 构建所有候选
        candidates = {}
        for stylizer in self.stylizers.values():
            candidate = stylizer.stylize(ctx.image_f32)
            candidates[candidate.style_id] = candidate
        
        ctx.set_cache(cache_key, candidates)
        return candidates


def load_pipeline(config_path: str | Path | None = None) -> CartoonPipeline:
    """
    便捷函数：加载 Pipeline
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        CartoonPipeline 实例
    """
    return CartoonPipeline(config_path)

