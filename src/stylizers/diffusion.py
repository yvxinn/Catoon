"""
DiffusionStylizer - 基于 Stable Diffusion + ControlNet 的风格化器

职责：
- 使用传统 toon base 作为 init_image，锁定色块布局
- 使用 Canny/XDoG 边缘作为 ControlNet 条件，锁定结构
- 生成对齐的多风格候选，供语义路由与融合使用

设计要点：
- 懒加载 pipeline，避免未安装 diffusers/权重缺失时报错
- 允许 fallback：若依赖不可用，则回退为“传统图 + 轻度线稿”以保证流程可跑
- 使用 Context 缓存避免重复推理
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from ..context import Context, StyleCandidate
from .base import BaseStylizer

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    from diffusers.schedulers import DDIMScheduler, EulerAncestralDiscreteScheduler

    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False


class DiffusionStylizer:
    """Stable Diffusion + ControlNet 风格化器"""

    def __init__(self, cfg: DictConfig):
        self.cfg = getattr(cfg, "diffusion", {})
        self.enabled: bool = bool(getattr(self.cfg, "enabled", False))
        self.model_id: str = getattr(self.cfg, "model", "runwayml/stable-diffusion-v1-5")
        self.controlnet_id: str | None = getattr(
            self.cfg, "controlnet_model", "lllyasviel/sd-controlnet-canny"
        )
        self.scheduler_name: str = getattr(self.cfg, "scheduler", "ddim")
        self.dtype: str = getattr(self.cfg, "dtype", "fp16")
        self.guidance_scale: float = float(getattr(self.cfg, "guidance_scale", 7.5))
        self.num_inference_steps: int = int(getattr(self.cfg, "num_inference_steps", 20))
        self.denoising_strength: float = float(
            getattr(self.cfg, "denoising_strength", 0.45)
        )
        self.controlnet_conditioning_scale: float = float(
            getattr(self.cfg, "controlnet_conditioning_scale", 0.9)
        )
        self.styles_cfg = list(getattr(self.cfg, "styles", []))
        self.device = self._resolve_device(cfg)

        # pipeline 相关
        self._pipe = None

    def generate_candidates(
        self,
        ctx: Context,
        traditional_image: np.ndarray,
        edge_map: np.ndarray,
        styles: Iterable[str] | None = None,
    ) -> dict[str, StyleCandidate]:
        """
        生成多风格 Diffusion 候选

        Args:
            ctx: 全局上下文，用于缓存
            traditional_image: 传统风格化输出 (float32 [0,1])
            edge_map: 边缘约束 (float32 [0,1])
            styles: 需要生成的风格 ID 列表；若为 None 则使用配置内全部风格
        """
        if not self.enabled:
            return {}

        selected_styles = self._filter_styles(styles)
        if not selected_styles:
            return {}

        cache_key = ctx.make_cache_key(
            "diffusion",
            self.model_id,
            self.controlnet_id or "none",
            f"denoise{self.denoising_strength:.2f}",
            f"ctrl{self.controlnet_conditioning_scale:.2f}",
            "styles_" + "_".join([s.get("name", "unnamed") for s in selected_styles]),
        )
        cached = ctx.get_cache(cache_key)
        if cached is not None:
            return cached

        self._ensure_pipeline()
        results: dict[str, StyleCandidate] = {}

        for style_cfg in selected_styles:
            style_id = style_cfg.get("name") or style_cfg.get("id") or style_cfg.get("style_id") or "Diffusion"
            try:
                if self._pipe is None:
                    result_img = self._fallback_image(traditional_image, edge_map)
                else:
                    result_img = self._run_diffusion(traditional_image, edge_map, style_cfg)
            except Exception as e:
                print(f"[DiffusionStylizer] {style_id} 推理失败，使用 fallback: {e}")
                result_img = self._fallback_image(traditional_image, edge_map)

            color_stats = BaseStylizer.compute_color_stats(result_img)
            results[style_id] = StyleCandidate(
                style_id=style_id,
                image=result_img,
                color_stats=color_stats,
                model_type="diffusion",
                model_name=self.model_id,
            )

        ctx.set_cache(cache_key, results)
        return results

    # ==================== 内部工具 ====================

    def _resolve_device(self, cfg: DictConfig) -> Any:
        """解析设备；若 torch 不可用则返回 cpu 字符串"""
        device_cfg = getattr(getattr(cfg, "global", {}), "device", "auto")
        if not _HAS_TORCH:
            return "cpu"
        if device_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_cfg)

    def _ensure_pipeline(self) -> None:
        """懒加载 Diffusers pipeline"""
        if self._pipe is not None:
            return
        if not self.enabled or not (_HAS_DIFFUSERS and _HAS_TORCH):
            print("[DiffusionStylizer] diffusers/torch 不可用，将使用 fallback。")
            self._pipe = None
            return
        if self.controlnet_id is None:
            print("[DiffusionStylizer] 未配置 controlnet_model，将使用 fallback。")
            self._pipe = None
            return

        try:
            torch_dtype = (
                torch.float16
                if self.dtype == "fp16" and getattr(self.device, "type", "cpu") != "cpu"
                else torch.float32
            )
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, torch_dtype=torch_dtype
            )
            # 使用 img2img + ControlNet pipeline（支持 init_image + control_image）
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )

            if self.scheduler_name == "euler_a":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config
                )
            elif self.scheduler_name == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            pipe.to(self.device)
            self._pipe = pipe
        except Exception as e:
            print(f"[DiffusionStylizer] Pipeline 初始化失败，将使用 fallback: {e}")
            self._pipe = None

    def _filter_styles(self, styles: Iterable[str] | None) -> list[dict]:
        """根据请求过滤需要生成的风格列表"""
        if styles is None:
            return list(self.styles_cfg)
        style_set = set(styles)
        return [s for s in self.styles_cfg if s.get("name") in style_set]

    def _run_diffusion(
        self,
        traditional_image: np.ndarray,
        edge_map: np.ndarray,
        style_cfg: dict,
    ) -> np.ndarray:
        """执行一次 ControlNet 引导的 img2img 扩散推理"""
        assert self._pipe is not None, "Pipeline 未初始化"

        init_pil = self._to_rgb_pil(traditional_image)
        control_pil = self._to_control_pil(edge_map)

        prompt = style_cfg.get("prompt") or style_cfg.get("name") or "anime style"
        negative_prompt = style_cfg.get("negative_prompt", "")
        guidance = float(style_cfg.get("guidance_scale", self.guidance_scale))
        steps = int(style_cfg.get("num_inference_steps", self.num_inference_steps))
        strength = float(style_cfg.get("denoising_strength", self.denoising_strength))
        ctrl_scale = float(
            style_cfg.get(
                "controlnet_conditioning_scale", self.controlnet_conditioning_scale
            )
        )

        with torch.no_grad():
            # StableDiffusionControlNetImg2ImgPipeline 参数：
            # - image: init_image (img2img 的起始图像)
            # - control_image: ControlNet 条件图像
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_pil,
                control_image=control_pil,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                controlnet_conditioning_scale=ctrl_scale,
            )

        result_pil = output.images[0]
        result = np.array(result_pil).astype(np.float32) / 255.0
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _to_rgb_pil(image: np.ndarray) -> Image.Image:
        """float32 [0,1] -> PIL RGB"""
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(image_u8, mode="RGB")

    @staticmethod
    def _to_control_pil(edge_map: np.ndarray) -> Image.Image:
        """float32 [0,1] 边缘图 -> PIL，ControlNet 输入"""
        edge_u8 = np.clip(edge_map * 255.0, 0, 255).astype(np.uint8)
        control = Image.fromarray(edge_u8, mode="L")
        return control.convert("RGB")

    @staticmethod
    def _fallback_image(
        traditional_image: np.ndarray,
        edge_map: np.ndarray,
        edge_strength: float = 0.3,
    ) -> np.ndarray:
        """
        轻量 fallback：在传统 toon base 上叠加柔和线稿，保证流程可跑。
        """
        edges = np.clip(edge_map, 0.0, 1.0)[..., None]
        result = traditional_image * (1 - edges * edge_strength)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

