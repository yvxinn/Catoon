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
        self._diffusion_stylizer = None
        self._region_stylizer = None  # 新增：区域级风格化器
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
    def diffusion_stylizer(self):
        """Diffusion 风格化器（懒加载）"""
        if self._diffusion_stylizer is None and hasattr(self.cfg, "diffusion"):
            try:
                from .stylizers import DiffusionStylizer
                self._diffusion_stylizer = DiffusionStylizer(self.cfg)
            except Exception as e:
                print(f"[Pipeline] DiffusionStylizer 初始化失败：{e}")
                self._diffusion_stylizer = None
        return self._diffusion_stylizer
    
    @property
    def region_stylizer(self):
        """区域级风格化器（懒加载）"""
        if self._region_stylizer is None:
            from .stylizers.region_stylizer import RegionStylizer
            self._region_stylizer = RegionStylizer(self.cfg, self.stylizers)
        return self._region_stylizer
    
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
        
        # C. 约束生成：结构 + 色彩
        edge_map = self._get_or_build_edge_map(ctx, ui_params)
        trad_candidate = self._get_or_build_traditional(ctx, ui_params)

        # C2. 风格候选生成（Diffusion 接管，含 Traditional）
        candidates = self._get_or_build_candidates(
            ctx=ctx,
            edge_map=edge_map,
            trad_candidate=trad_candidate,
            ui_params=ui_params
        )
        
        # D. 语义路由
        routing = self.router.route(
            semantic_masks=seg_out.semantic_masks,
            face_mask=face_mask,
            ui_overrides=ui_params
        )
        
        # 区域级风格化（按需生成，带缓存）
        region_candidates = None
        if ui_params.get("enable_region_k", True):
            region_candidates = self.region_stylizer.generate_region_styles(
                image_f32=ctx.image_f32,
                image_hash=ctx.image_hash,
                seg_out=seg_out,
                region_configs=routing.region_configs,
                global_candidates=candidates
            )
        
        # E. 区域融合（传递原图用于 strength 混合）
        fused = self.fuser.fuse(
            candidates=candidates,
            routing=routing,
            seg_out=seg_out,
            method=ui_params.get("fusion_method", self.cfg.fusion.default_method),
            blur_kernel=ui_params.get("fusion_blur_kernel"),
            original_image=ctx.image_f32,
            region_candidates=region_candidates
        )
        
        # F. 全局协调
        if ui_params.get("harmonization_enabled", self.cfg.harmonization.enabled):
            ref = self.harmonizer.pick_reference(
                candidates, seg_out, ui_params, self.cfg.harmonization
            )
            fused = self.harmonizer.match_and_adjust(fused, ref, ui_params)
        
        # G. 线稿叠加（复用前置 edge_map）
        overlay_edges = ui_params.get(
            "overlay_edges", getattr(self.cfg.lineart, "overlay_edges", True)
        )
        edge_strength = ui_params.get("edge_strength", self.cfg.lineart.default_strength)
        if overlay_edges and edge_strength > 1e-3:
            fused = self.lineart.overlay(fused, edge_map, edge_strength, ui_params)
        
        # G2. 细节增强 (Phase 3 Guided Filter)
        if ui_params.get("detail_enhance_enabled", False):
            detail_strength = ui_params.get("detail_strength", 0.5)
            fused = self.lineart.enhance_detail(fused, ctx.image_f32, detail_strength)
        
        # H. 深度增强（可选）
        if self.depth_enhancer and ui_params.get("depth_fog_enabled", False):
            depth_map = self.depth_enhancer.estimate(ctx.image_u8)
            fused = self.depth_enhancer.apply_fog(fused, depth_map, ui_params)
        
        # 后处理（恢复原始尺寸）
        out_u8 = self.preprocessor.postprocess(fused, ctx)
        return out_u8
    
    def _get_or_build_edge_map(
        self,
        ctx: Context,
        ui_params: dict | None = None
    ) -> np.ndarray:
        """生成或获取缓存的线稿（用于 ControlNet 与叠加）"""
        ui_params = ui_params or {}
        engine = ui_params.get("line_engine", self.cfg.lineart.engine)
        line_width = ui_params.get("line_width", self.cfg.lineart.line_width)
        canny_low = ui_params.get("canny_low", self.cfg.lineart.canny_low)
        canny_high = ui_params.get("canny_high", self.cfg.lineart.canny_high)
        xdog_sigma = ui_params.get("xdog_sigma", getattr(self.cfg.lineart, "xdog_sigma", 0.5))
        xdog_k = ui_params.get("xdog_k", getattr(self.cfg.lineart, "xdog_k", 1.6))
        xdog_p = ui_params.get("xdog_p", getattr(self.cfg.lineart, "xdog_p", 19.0))
        
        cache_key = ctx.make_cache_key(
            "edge_map",
            engine,
            f"w{line_width}",
            f"c{canny_low}_{canny_high}",
            f"xd{xdog_sigma:.2f}_{xdog_k:.2f}_{xdog_p:.1f}"
        )
        cached = ctx.get_cache(cache_key)
        if cached is not None:
            return cached
        
        edges = self.lineart.extract(ctx.image_u8, ui_params)
        ctx.set_cache(cache_key, edges)
        return edges
    
    def _get_or_build_traditional(
        self,
        ctx: Context,
        ui_params: dict | None = None
    ):
        """生成或获取缓存的传统 toon 候选（兼作 Diffusion init_image）"""
        ui_params = ui_params or {}
        trad_k = ui_params.get("traditional_k", self.cfg.stylizers.traditional.default_K)
        trad_method = ui_params.get(
            "traditional_smooth_method", self.cfg.stylizers.traditional.smooth_method
        )
        cache_key = ctx.make_cache_key(f"traditional_{trad_k}_{trad_method}")
        cached = ctx.get_cache(cache_key)
        if cached is not None:
            return cached
        
        stylizer = self.stylizers.get("Traditional")
        if stylizer is None:
            raise RuntimeError("Traditional stylizer is required for diffusion pipeline.")
        
        candidate = stylizer.stylize(
            ctx.image_f32,
            K=trad_k,
            smooth_method=trad_method
        )
        ctx.set_cache(cache_key, candidate)
        return candidate
    
    def _build_gan_candidates(self, ctx: Context) -> dict[str, Any]:
        """可选 GAN 候选（用于回退或对比）"""
        candidates: dict[str, Any] = {}
        for stylizer in self.stylizers.values():
            if getattr(stylizer, "model_type", "") == "gan":
                try:
                    candidate = stylizer.stylize(ctx.image_f32)
                    candidates[candidate.style_id] = candidate
                except Exception as e:
                    print(f"[Pipeline] GAN stylizer {getattr(stylizer, 'style_id', '')} failed: {e}")
        return candidates
    
    def _get_or_build_candidates(
        self,
        ctx: Context,
        edge_map: np.ndarray,
        trad_candidate,
        ui_params: dict | None = None
    ) -> dict:
        """获取或构建风格候选（Diffusion + Traditional + 可选 GAN）"""
        ui_params = ui_params or {}
        
        trad_k = ui_params.get("traditional_k", self.cfg.stylizers.traditional.default_K)
        trad_method = ui_params.get(
            "traditional_smooth_method", self.cfg.stylizers.traditional.smooth_method
        )
        diff_cfg = getattr(self.cfg, "diffusion", {})
        diff_enabled = bool(getattr(diff_cfg, "enabled", False))
        
        style_names: list[str] = []
        for s in getattr(diff_cfg, "styles", []):
            name = getattr(s, "name", None) or (s.get("name") if isinstance(s, dict) else None)
            if name:
                style_names.append(name)
        
        cache_key = ctx.make_cache_key(
            "candidates_v2",
            f"k{trad_k}",
            trad_method,
            "diff_on" if diff_enabled else "diff_off",
            "styles_" + "_".join(style_names)
        )
        cached = ctx.get_cache(cache_key)
        if cached is not None:
            return cached
        
        candidates: dict[str, Any] = {}
        
        # Diffusion 候选
        if diff_enabled and self.diffusion_stylizer is not None:
            diff_candidates = self.diffusion_stylizer.generate_candidates(
                ctx=ctx,
                traditional_image=trad_candidate.image,
                edge_map=edge_map,
                styles=style_names or None
            )
            candidates.update(diff_candidates)
        
        # 可选 GAN 候选（回退或对比）
        use_gan = ui_params.get("enable_gan_candidates", self.cfg.stylizers.get("enable_gan", False))
        if use_gan:
            candidates.update(self._build_gan_candidates(ctx))
        
        # 保留传统候选
        candidates["Traditional"] = trad_candidate
        
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

