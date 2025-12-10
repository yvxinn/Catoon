"""
Harmonizer - 全局协调器

使用直方图匹配和色调控制解决"缝合怪"问题。
"""

import cv2
import numpy as np
from skimage.exposure import match_histograms
from omegaconf import DictConfig

from ..context import SegmentationOutput, StyleCandidate


class Harmonizer:
    """全局协调器"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化协调器
        
        Args:
            cfg: 配置对象
        """
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
        """
        选择参考图像用于直方图匹配
        
        Args:
            candidates: 候选字典
            seg_out: 分割输出
            ui_params: UI 参数
            harmo_cfg: 协调配置
        
        Returns:
            参考图像 float32 (H,W,3)
        """
        reference_mode = ui_params.get(
            "harmonization_reference",
            self.reference_region
        )
        
        if reference_mode == "auto":
            # 使用面积最大的区域
            max_area = 0
            best_region = "SKY"
            
            for region_name, mask in seg_out.semantic_masks.items():
                area = mask.sum()
                if area > max_area:
                    max_area = area
                    best_region = region_name
            
            reference_mode = best_region
        
        # 获取对应区域的风格候选
        # 这里简化处理，直接使用第一个可用的候选
        if reference_mode in candidates:
            return candidates[reference_mode].image
        
        # 默认使用 Traditional 或第一个可用的候选
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
        执行直方图匹配和色调调整
        
        Args:
            fused: 融合后的图像 float32
            reference: 参考图像 float32
            ui_params: UI 参数
        
        Returns:
            协调后的图像 float32
        """
        # 获取匹配强度
        strength = ui_params.get(
            "harmonization_strength",
            self.default_strength
        )
        
        if strength < 0.01:
            # 不进行匹配
            result = fused
        else:
            # 转换为 uint8 进行直方图匹配
            fused_u8 = (fused * 255).astype(np.uint8)
            ref_u8 = (reference * 255).astype(np.uint8)
            
            # 执行直方图匹配
            matched_u8 = match_histograms(
                fused_u8, ref_u8, channel_axis=-1
            )
            
            # 转回 float32
            matched = matched_u8.astype(np.float32) / 255.0
            
            # 按强度混合
            result = fused * (1 - strength) + matched * strength
        
        # 应用全局色调调整
        result = self._apply_tone_adjustments(result, ui_params)
        
        return np.clip(result, 0, 1)
    
    def _apply_tone_adjustments(
        self,
        image: np.ndarray,
        ui_params: dict
    ) -> np.ndarray:
        """
        应用全局色调调整
        
        Args:
            image: 输入图像 float32
            ui_params: UI 参数
        
        Returns:
            调整后的图像
        """
        result = image.copy()
        
        # Gamma 调整
        gamma = ui_params.get("gamma", 1.0)
        if abs(gamma - 1.0) > 0.01:
            result = np.power(result, 1.0 / gamma)
        
        # 对比度调整
        contrast = ui_params.get("contrast", 1.0)
        if abs(contrast - 1.0) > 0.01:
            mean = result.mean()
            result = (result - mean) * contrast + mean
        
        # 饱和度调整
        saturation = ui_params.get("saturation", 1.0)
        if abs(saturation - 1.0) > 0.01:
            # 转换到 HSV
            result_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(result_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            result_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = result_u8.astype(np.float32) / 255.0
        
        # 亮度调整
        brightness = ui_params.get("brightness", 0.0)
        if abs(brightness) > 0.01:
            result = result + brightness / 255.0
        
        return result


def create_harmonizer(cfg: DictConfig) -> Harmonizer:
    """
    便捷函数：创建协调器
    
    Args:
        cfg: 配置对象
    
    Returns:
        Harmonizer 实例
    """
    return Harmonizer(cfg)

