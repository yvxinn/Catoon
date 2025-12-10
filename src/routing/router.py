"""
SemanticRouter - 语义路由器

将语义分割结果映射到风格配置，并处理人脸保护策略。
"""

import numpy as np
from omegaconf import DictConfig

from ..context import RoutingPlan, RegionConfig
from ..segmentation.bucket_mapper import SemanticBucket


class SemanticRouter:
    """语义路由器"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化路由器
        
        Args:
            cfg: 配置对象
        """
        self.cfg = cfg
        self.region_presets = cfg.region_presets
        self.face_policy = cfg.face_policy
    
    def route(
        self,
        semantic_masks: dict[str, np.ndarray],
        face_mask: np.ndarray | None = None,
        ui_overrides: dict | None = None
    ) -> RoutingPlan:
        """
        生成路由计划
        
        Args:
            semantic_masks: {bucket_name: mask} 语义掩码字典
            face_mask: 人脸掩码，可选
            ui_overrides: UI 参数覆盖，可选
        
        Returns:
            RoutingPlan 路由计划
        """
        ui_overrides = ui_overrides or {}
        region_overrides = ui_overrides.get("region_overrides", {})
        
        # 构建每个区域的配置
        region_configs = {}
        
        for bucket in SemanticBucket:
            bucket_name = bucket.value
            
            # 从预设获取默认配置
            preset = self.region_presets.get(bucket_name, {})
            
            # 应用 UI 覆盖
            override = region_overrides.get(bucket_name, {})
            
            # 合并配置
            style_id = override.get("style", preset.get("style", "Traditional"))
            mix_weight = override.get("weight", preset.get("mix_weight", 1.0))
            
            # 获取 toon 参数
            toon_params = preset.get("toon_params", {})
            toon_K = toon_params.get("K", 16)
            smooth_strength = toon_params.get("smooth_strength", 0.6)
            edge_strength = toon_params.get("edge_strength", 0.5)
            
            region_configs[bucket_name] = RegionConfig(
                style_id=style_id,
                mix_weight=mix_weight,
                toon_K=toon_K,
                smooth_strength=smooth_strength,
                edge_strength=edge_strength
            )
        
        # 处理人脸保护
        face_protection_mask = None
        face_policy_mode = "protect"
        
        if self.face_policy.enabled and face_mask is not None:
            face_protect_enabled = ui_overrides.get(
                "face_protect_enabled", 
                self.face_policy.enabled
            )
            
            if face_protect_enabled:
                face_protection_mask = face_mask
                face_policy_mode = ui_overrides.get(
                    "face_protect_mode",
                    self.face_policy.mode
                )
                
                # 应用人脸保护策略到 PERSON 区域
                self._apply_face_protection(
                    region_configs, 
                    face_policy_mode,
                    ui_overrides
                )
        
        return RoutingPlan(
            region_configs=region_configs,
            face_protection_mask=face_protection_mask,
            face_policy_mode=face_policy_mode
        )
    
    def _apply_face_protection(
        self,
        region_configs: dict[str, RegionConfig],
        mode: str,
        ui_overrides: dict
    ) -> None:
        """
        应用人脸保护策略
        
        Args:
            region_configs: 区域配置字典（会被修改）
            mode: 保护模式 "protect" | "blend" | "full_style"
            ui_overrides: UI 覆盖参数
        """
        if mode == "full_style":
            # 不保护，使用原始配置
            return
        
        person_config = region_configs.get("PERSON")
        if person_config is None:
            return
        
        if mode == "protect":
            # 强制使用传统风格，限制 GAN 权重
            face_style = self.face_policy.get("face_style", "Traditional")
            gan_weight_max = ui_overrides.get(
                "face_gan_weight_max",
                self.face_policy.get("gan_weight_max", 0.3)
            )
            
            # 更新 PERSON 配置
            region_configs["PERSON"] = RegionConfig(
                style_id=face_style,
                mix_weight=min(person_config.mix_weight, gan_weight_max),
                toon_K=person_config.toon_K,
                smooth_strength=person_config.smooth_strength,
                edge_strength=person_config.edge_strength
            )
        
        elif mode == "blend":
            # 混合模式：降低风格权重
            gan_weight_max = ui_overrides.get(
                "face_gan_weight_max",
                self.face_policy.get("gan_weight_max", 0.3)
            )
            
            region_configs["PERSON"] = RegionConfig(
                style_id=person_config.style_id,
                mix_weight=min(person_config.mix_weight, gan_weight_max + 0.3),
                toon_K=person_config.toon_K,
                smooth_strength=person_config.smooth_strength * 0.7,  # 降低平滑
                edge_strength=person_config.edge_strength
            )
    
    def get_style_for_region(
        self, 
        region_name: str,
        routing_plan: RoutingPlan
    ) -> tuple[str, float]:
        """
        获取指定区域的风格配置
        
        Args:
            region_name: 区域名称
            routing_plan: 路由计划
        
        Returns:
            (style_id, mix_weight) 元组
        """
        config = routing_plan.region_configs.get(region_name)
        if config is None:
            return "Traditional", 1.0
        return config.style_id, config.mix_weight
    
    def get_available_styles(self) -> list[str]:
        """
        获取所有可用的风格 ID
        
        Returns:
            风格 ID 列表
        """
        styles = set()
        
        # 从区域预设收集
        for bucket_name, preset in self.region_presets.items():
            style = preset.get("style")
            if style:
                styles.add(style)
        
        # 添加默认风格
        styles.add("Traditional")
        
        return sorted(list(styles))


def create_router(cfg: DictConfig) -> SemanticRouter:
    """
    便捷函数：创建路由器
    
    Args:
        cfg: 配置对象
    
    Returns:
        SemanticRouter 实例
    """
    return SemanticRouter(cfg)

