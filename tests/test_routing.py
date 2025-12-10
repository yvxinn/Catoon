"""
Routing 模块单元测试
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.routing import SemanticRouter
from src.context import RoutingPlan, RegionConfig
from src.segmentation.bucket_mapper import SemanticBucket


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {
            "max_image_size": 512,
            "device": "cpu"
        },
        "region_presets": {
            "SKY": {
                "style": "Shinkai",
                "mix_weight": 1.0,
                "toon_params": {"K": 8, "smooth_strength": 0.7, "edge_strength": 0.3}
            },
            "PERSON": {
                "style": "Hayao",
                "mix_weight": 1.0,
                "toon_params": {"K": 24, "smooth_strength": 0.4, "edge_strength": 0.6}
            },
            "BUILDING": {
                "style": "Traditional",
                "mix_weight": 1.0,
                "toon_params": {"K": 16, "smooth_strength": 0.6, "edge_strength": 0.5}
            },
            "VEGETATION": {
                "style": "Paprika",
                "mix_weight": 0.8,
                "toon_params": {"K": 16, "smooth_strength": 0.5, "edge_strength": 0.5}
            },
            "ROAD": {
                "style": "Traditional",
                "mix_weight": 1.0,
                "toon_params": {"K": 12, "smooth_strength": 0.6, "edge_strength": 0.4}
            },
            "WATER": {
                "style": "Shinkai",
                "mix_weight": 0.9,
                "toon_params": {"K": 10, "smooth_strength": 0.8, "edge_strength": 0.2}
            },
            "OTHERS": {
                "style": "Traditional",
                "mix_weight": 1.0,
                "toon_params": {"K": 16, "smooth_strength": 0.6, "edge_strength": 0.5}
            }
        },
        "face_policy": {
            "enabled": True,
            "mode": "protect",
            "face_style": "Traditional",
            "gan_weight_max": 0.3,
            "preserve_skin_tone": True
        }
    })


@pytest.fixture
def semantic_masks():
    """创建测试语义掩码"""
    h, w = 256, 256
    masks = {}
    
    for bucket in SemanticBucket:
        # 创建简单的测试 mask
        mask = np.zeros((h, w), dtype=np.float32)
        masks[bucket.value] = mask
    
    # 设置一些区域
    masks["SKY"][:64, :] = 1.0  # 上方是天空
    masks["BUILDING"][64:192, :128] = 1.0  # 左侧是建筑
    masks["VEGETATION"][64:192, 128:] = 1.0  # 右侧是植被
    masks["ROAD"][192:, :] = 1.0  # 下方是道路
    
    return masks


@pytest.fixture
def face_mask():
    """创建测试人脸掩码"""
    h, w = 256, 256
    mask = np.zeros((h, w), dtype=np.float32)
    # 中心有一个人脸区域
    mask[80:150, 100:156] = 1.0
    return mask


class TestSemanticRouter:
    """SemanticRouter 测试类"""
    
    def test_init(self, config):
        """测试初始化"""
        router = SemanticRouter(config)
        
        assert router.region_presets is not None
        assert router.face_policy is not None
    
    def test_route_returns_routing_plan(self, config, semantic_masks):
        """测试路由返回 RoutingPlan"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks)
        
        assert isinstance(plan, RoutingPlan)
    
    def test_route_contains_all_regions(self, config, semantic_masks):
        """测试路由包含所有区域"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks)
        
        for bucket in SemanticBucket:
            assert bucket.value in plan.region_configs
    
    def test_route_uses_preset_styles(self, config, semantic_masks):
        """测试使用预设风格"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks)
        
        assert plan.region_configs["SKY"].style_id == "Shinkai"
        assert plan.region_configs["PERSON"].style_id == "Hayao"
        assert plan.region_configs["BUILDING"].style_id == "Traditional"
    
    def test_route_uses_preset_weights(self, config, semantic_masks):
        """测试使用预设权重"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks)
        
        assert plan.region_configs["SKY"].mix_weight == 1.0
        assert plan.region_configs["VEGETATION"].mix_weight == 0.8
    
    def test_route_uses_toon_params(self, config, semantic_masks):
        """测试使用 toon 参数"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks)
        
        sky_config = plan.region_configs["SKY"]
        assert sky_config.toon_K == 8
        assert sky_config.smooth_strength == 0.7
        assert sky_config.edge_strength == 0.3


class TestUIOverrides:
    """UI 覆盖参数测试"""
    
    def test_override_style(self, config, semantic_masks):
        """测试覆盖风格"""
        router = SemanticRouter(config)
        
        ui_overrides = {
            "region_overrides": {
                "SKY": {"style": "Traditional"}
            }
        }
        
        plan = router.route(semantic_masks, ui_overrides=ui_overrides)
        
        assert plan.region_configs["SKY"].style_id == "Traditional"
    
    def test_override_weight(self, config, semantic_masks):
        """测试覆盖权重"""
        router = SemanticRouter(config)
        
        ui_overrides = {
            "region_overrides": {
                "VEGETATION": {"weight": 0.5}
            }
        }
        
        plan = router.route(semantic_masks, ui_overrides=ui_overrides)
        
        assert plan.region_configs["VEGETATION"].mix_weight == 0.5
    
    def test_override_multiple_regions(self, config, semantic_masks):
        """测试覆盖多个区域"""
        router = SemanticRouter(config)
        
        ui_overrides = {
            "region_overrides": {
                "SKY": {"style": "Paprika", "weight": 0.9},
                "BUILDING": {"style": "Shinkai", "weight": 0.7}
            }
        }
        
        plan = router.route(semantic_masks, ui_overrides=ui_overrides)
        
        assert plan.region_configs["SKY"].style_id == "Paprika"
        assert plan.region_configs["SKY"].mix_weight == 0.9
        assert plan.region_configs["BUILDING"].style_id == "Shinkai"
        assert plan.region_configs["BUILDING"].mix_weight == 0.7


class TestFaceProtection:
    """人脸保护测试"""
    
    def test_no_face_mask(self, config, semantic_masks):
        """测试无人脸掩码"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks, face_mask=None)
        
        assert plan.face_protection_mask is None
    
    def test_face_mask_stored(self, config, semantic_masks, face_mask):
        """测试人脸掩码被存储"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks, face_mask=face_mask)
        
        assert plan.face_protection_mask is not None
        assert plan.face_protection_mask.shape == face_mask.shape
    
    def test_protect_mode_changes_person_style(self, config, semantic_masks, face_mask):
        """测试 protect 模式改变 PERSON 风格"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks, face_mask=face_mask)
        
        # protect 模式应该将 PERSON 改为 Traditional
        assert plan.region_configs["PERSON"].style_id == "Traditional"
        assert plan.face_policy_mode == "protect"
    
    def test_protect_mode_limits_weight(self, config, semantic_masks, face_mask):
        """测试 protect 模式限制权重"""
        router = SemanticRouter(config)
        
        plan = router.route(semantic_masks, face_mask=face_mask)
        
        # 权重应该被限制到 gan_weight_max
        assert plan.region_configs["PERSON"].mix_weight <= 0.3
    
    def test_full_style_mode(self, config, semantic_masks, face_mask):
        """测试 full_style 模式不保护"""
        router = SemanticRouter(config)
        
        ui_overrides = {
            "face_protect_mode": "full_style"
        }
        
        plan = router.route(semantic_masks, face_mask=face_mask, ui_overrides=ui_overrides)
        
        # full_style 模式应该保持原始风格
        assert plan.region_configs["PERSON"].style_id == "Hayao"
    
    def test_disabled_face_protection(self, config, semantic_masks, face_mask):
        """测试禁用人脸保护"""
        router = SemanticRouter(config)
        
        ui_overrides = {
            "face_protect_enabled": False
        }
        
        plan = router.route(semantic_masks, face_mask=face_mask, ui_overrides=ui_overrides)
        
        assert plan.face_protection_mask is None


class TestHelperMethods:
    """辅助方法测试"""
    
    def test_get_style_for_region(self, config, semantic_masks):
        """测试获取区域风格"""
        router = SemanticRouter(config)
        plan = router.route(semantic_masks)
        
        style, weight = router.get_style_for_region("SKY", plan)
        
        assert style == "Shinkai"
        assert weight == 1.0
    
    def test_get_style_for_unknown_region(self, config, semantic_masks):
        """测试获取未知区域风格"""
        router = SemanticRouter(config)
        plan = router.route(semantic_masks)
        
        style, weight = router.get_style_for_region("UNKNOWN", plan)
        
        assert style == "Traditional"
        assert weight == 1.0
    
    def test_get_available_styles(self, config):
        """测试获取可用风格"""
        router = SemanticRouter(config)
        
        styles = router.get_available_styles()
        
        assert "Traditional" in styles
        assert "Shinkai" in styles
        assert "Hayao" in styles
        assert "Paprika" in styles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

