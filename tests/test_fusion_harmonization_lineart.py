"""
Fusion, Harmonization, Lineart 模块测试
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.fusion import SoftMaskFusion, FusionModule
from src.harmonization import Harmonizer
from src.lineart import CannyLineart, LineartEngine
from src.context import StyleCandidate, ColorStats, RoutingPlan, RegionConfig, SegmentationOutput
from src.segmentation.bucket_mapper import SemanticBucket


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {"max_image_size": 512, "device": "cpu"},
        "fusion": {
            "default_method": "soft_mask",
            "soft_mask_blur_kernel": 21,
            "pyramid_levels": 6,
            "enable_seamless_blend": False,
            "poisson_roi": "boundary_band"
        },
        "harmonization": {
            "enabled": True,
            "reference_region": "SKY",
            "histogram_matching": True,
            "match_strength": 0.8
        },
        "lineart": {
            "engine": "canny",
            "default_strength": 0.5,
            "canny_low": 100,
            "canny_high": 200,
            "line_width": 1
        }
    })


@pytest.fixture
def sample_image():
    """测试图像"""
    return np.random.rand(128, 128, 3).astype(np.float32)


@pytest.fixture
def sample_image_u8():
    """测试图像 uint8"""
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


def create_mock_candidate(style_id: str, shape: tuple) -> StyleCandidate:
    """创建模拟的风格候选"""
    image = np.random.rand(*shape).astype(np.float32)
    stats = ColorStats(
        lab_mean=np.array([128.0, 0.0, 0.0]),
        lab_std=np.array([50.0, 20.0, 20.0])
    )
    return StyleCandidate(
        style_id=style_id,
        image=image,
        color_stats=stats,
        model_type="test"
    )


def create_mock_seg_output(shape: tuple) -> SegmentationOutput:
    """创建模拟的分割输出"""
    h, w = shape[:2]
    masks = {}
    for bucket in SemanticBucket:
        mask = np.zeros((h, w), dtype=np.float32)
        masks[bucket.value] = mask
    
    # 设置一些区域
    masks["SKY"][:h//2, :] = 1.0
    masks["BUILDING"][h//2:, :w//2] = 1.0
    masks["VEGETATION"][h//2:, w//2:] = 1.0
    
    return SegmentationOutput(
        label_map=np.zeros((h, w), dtype=np.int32),
        semantic_masks=masks
    )


def create_mock_routing() -> RoutingPlan:
    """创建模拟的路由计划"""
    configs = {}
    for bucket in SemanticBucket:
        configs[bucket.value] = RegionConfig(
            style_id="Traditional",
            mix_weight=1.0
        )
    configs["SKY"].style_id = "Shinkai"
    
    return RoutingPlan(
        region_configs=configs,
        face_protection_mask=None
    )


# ========== Fusion 测试 ==========

class TestSoftMaskFusion:
    """SoftMaskFusion 测试"""
    
    def test_init(self):
        """测试初始化"""
        fuser = SoftMaskFusion(blur_kernel=21)
        assert fuser.blur_kernel == 21
    
    def test_fuse_output_shape(self, sample_image):
        """测试输出形状"""
        fuser = SoftMaskFusion()
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape),
            "Shinkai": create_mock_candidate("Shinkai", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = fuser.fuse(candidates, routing, seg_out)
        
        assert result.shape == sample_image.shape
        assert result.dtype == np.float32
    
    def test_fuse_output_range(self, sample_image):
        """测试输出范围"""
        fuser = SoftMaskFusion()
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = fuser.fuse(candidates, routing, seg_out)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestFusionModule:
    """FusionModule 测试"""
    
    def test_init(self, config):
        """测试初始化"""
        module = FusionModule(config)
        assert module.default_method == "soft_mask"
    
    def test_fuse_soft_mask(self, config, sample_image):
        """测试 soft_mask 方法"""
        module = FusionModule(config)
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = module.fuse(candidates, routing, seg_out, method="soft_mask")
        
        assert result.shape == sample_image.shape


# ========== Harmonization 测试 ==========

class TestHarmonizer:
    """Harmonizer 测试"""
    
    def test_init(self, config):
        """测试初始化"""
        harmonizer = Harmonizer(config)
        assert harmonizer.enabled == True
        assert harmonizer.reference_region == "SKY"
    
    def test_pick_reference(self, config, sample_image):
        """测试选择参考图像"""
        harmonizer = Harmonizer(config)
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape),
            "Shinkai": create_mock_candidate("Shinkai", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        
        ref = harmonizer.pick_reference(candidates, seg_out, {}, config.harmonization)
        
        assert ref.shape == sample_image.shape
    
    def test_match_and_adjust(self, config, sample_image):
        """测试匹配和调整"""
        harmonizer = Harmonizer(config)
        
        reference = np.random.rand(*sample_image.shape).astype(np.float32)
        
        result = harmonizer.match_and_adjust(sample_image, reference, {})
        
        assert result.shape == sample_image.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_gamma_adjustment(self, config, sample_image):
        """测试 gamma 调整"""
        harmonizer = Harmonizer(config)
        reference = sample_image.copy()
        
        # 高 gamma 应该让图像变亮
        result_high = harmonizer.match_and_adjust(
            sample_image, reference, {"gamma": 2.0, "harmonization_strength": 0}
        )
        
        # 低 gamma 应该让图像变暗
        result_low = harmonizer.match_and_adjust(
            sample_image, reference, {"gamma": 0.5, "harmonization_strength": 0}
        )
        
        assert result_high.mean() > result_low.mean()


# ========== Lineart 测试 ==========

class TestCannyLineart:
    """CannyLineart 测试"""
    
    def test_init(self):
        """测试初始化"""
        lineart = CannyLineart(low_threshold=100, high_threshold=200)
        assert lineart.low_threshold == 100
        assert lineart.high_threshold == 200
    
    def test_extract_output_shape(self, sample_image_u8):
        """测试提取输出形状"""
        lineart = CannyLineart()
        
        edges = lineart.extract(sample_image_u8)
        
        assert edges.shape == sample_image_u8.shape[:2]
        assert edges.dtype == np.float32
    
    def test_extract_output_range(self, sample_image_u8):
        """测试提取输出范围"""
        lineart = CannyLineart()
        
        edges = lineart.extract(sample_image_u8)
        
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0
    
    def test_overlay_output_shape(self, sample_image):
        """测试叠加输出形状"""
        lineart = CannyLineart()
        
        edges = np.random.rand(128, 128).astype(np.float32)
        
        result = lineart.overlay(sample_image, edges, strength=0.5)
        
        assert result.shape == sample_image.shape
    
    def test_overlay_strength_zero(self, sample_image):
        """测试零强度叠加"""
        lineart = CannyLineart()
        
        edges = np.ones((128, 128), dtype=np.float32)
        
        result = lineart.overlay(sample_image, edges, strength=0.0)
        
        np.testing.assert_array_almost_equal(result, sample_image)
    
    def test_overlay_darkens_edges(self, sample_image):
        """测试边缘区域变暗"""
        lineart = CannyLineart()
        
        # 创建一个简单的边缘
        edges = np.zeros((128, 128), dtype=np.float32)
        edges[64, :] = 1.0  # 水平线
        
        result = lineart.overlay(sample_image, edges, strength=1.0)
        
        # 边缘处应该变暗（乘以 0）
        assert result[64, 0].mean() < sample_image[64, 0].mean()


class TestLineartEngine:
    """LineartEngine 测试"""
    
    def test_init(self, config):
        """测试初始化"""
        engine = LineartEngine(config)
        assert engine.default_engine == "canny"
    
    def test_extract(self, config, sample_image_u8):
        """测试提取"""
        engine = LineartEngine(config)
        
        edges = engine.extract(sample_image_u8, {})
        
        assert edges.shape == sample_image_u8.shape[:2]
    
    def test_overlay(self, config, sample_image):
        """测试叠加"""
        engine = LineartEngine(config)
        
        edges = np.random.rand(128, 128).astype(np.float32)
        
        result = engine.overlay(sample_image, edges, 0.5, {})
        
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

