"""
Harmonizer - 全局协调器

使用直方图匹配解决"缝合怪"问题。
"""

import cv2
import numpy as np
from skimage.exposure import match_histograms
from omegaconf import DictConfig

from ..context import SegmentationOutput, StyleCandidate


class Harmonizer:
    """全局协调器"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.harmonization
        self.enabled = self.cfg.enabled
        self.reference_region = self.cfg.reference_region
        self.default_strength = self.cfg.get("match_strength", 0.8)
    
    def pick_reference(
        self,
        candidates: dict[str, StyleCandidate],
        seg_out: SegmentationOutput,
        ui_params: dict,
        harmo_cfg: DictConfig
    ) -> np.ndarray:
        """选择参考图像用于直方图匹配"""
        reference_mode = ui_params.get("harmonization_reference", self.reference_region)
        
        if reference_mode == "auto":
            max_area = 0
            best_region = "SKY"
            for region_name, mask in seg_out.semantic_masks.items():
                area = mask.sum()
                if area > max_area:
                    max_area = area
                    best_region = region_name
            reference_mode = best_region
        
        if reference_mode in candidates:
            return candidates[reference_mode].image
        
        if "Traditional" in candidates:
            return candidates["Traditional"].image
        
        return next(iter(candidates.values())).image
    
    def match_and_adjust(
        self,
        fused: np.ndarray,
        reference: np.ndarray,
        ui_params: dict
    ) -> np.ndarray:
        """
        执行直方图匹配
        
        注意：色调调整（gamma、对比度、饱和度、亮度）已移至 ui/logic.py 的 apply_tone_adjustment
        """
        strength = ui_params.get("harmonization_strength", self.default_strength)
        
        if strength < 0.01:
            return fused
        
        fused_u8 = (fused * 255).astype(np.uint8)
        ref_u8 = (reference * 255).astype(np.uint8)
        
        matched_u8 = match_histograms(fused_u8, ref_u8, channel_axis=-1)
        matched = matched_u8.astype(np.float32) / 255.0
        
        result = fused * (1 - strength) + matched * strength
        return np.clip(result, 0, 1)


def create_harmonizer(cfg: DictConfig) -> Harmonizer:
    """便捷函数：创建协调器"""
    return Harmonizer(cfg)

