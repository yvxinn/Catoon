"""
Phase 2 测试 - AnimeGAN 和 Pyramid 融合
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.stylizers import AnimeGANStylizer, TraditionalStylizer
from src.fusion import LaplacianPyramidFusion, FusionModule
from src.context import StyleCandidate, ColorStats, RoutingPlan, RegionConfig, SegmentationOutput
from src.segmentation.bucket_mapper import SemanticBucket


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {"max_image_size": 512, "device": "cpu"},
        "stylizers": {
            "gan": [
                {"name": "Hayao", "model": "AnimeGANv2", "weights": "Hayao"},
                {"name": "Shinkai", "model": "AnimeGANv2", "weights": "Shinkai"},
            ],
            "traditional": {
                "smooth_method": "bilateral",
                "default_K": 16,
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75
            }
        },
        "fusion": {
            "default_method": "laplacian_pyramid",
            "soft_mask_blur_kernel": 21,
            "pyramid_levels": 4
        }
    })


@pytest.fixture
def sample_image():
    """测试图像"""
    return np.random.rand(128, 128, 3).astype(np.float32)


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
    configs["VEGETATION"].style_id = "Hayao"
    
    return RoutingPlan(
        region_configs=configs,
        face_protection_mask=None
    )


# ========== AnimeGAN 测试 ==========

class TestAnimeGANStylizer:
    """AnimeGAN 风格化器测试"""
    
    def test_available_styles(self):
        """测试可用风格列表"""
        styles = AnimeGANStylizer.get_available_styles()
        assert "Hayao" in styles
        assert "Shinkai" in styles
        assert "Paprika" in styles
    
    def test_init(self, config):
        """测试初始化"""
        stylizer = AnimeGANStylizer("Hayao", config)
        assert stylizer.style_id == "Hayao"
        assert stylizer.model_type == "gan"
    
    def test_stylize_returns_candidate(self, config, sample_image):
        """测试风格化返回候选"""
        stylizer = AnimeGANStylizer("Shinkai", config)
        result = stylizer.stylize(sample_image)
        
        assert isinstance(result, StyleCandidate)
        assert result.style_id == "Shinkai"
    
    def test_stylize_output_shape(self, config, sample_image):
        """测试输出形状"""
        stylizer = AnimeGANStylizer("Hayao", config)
        result = stylizer.stylize(sample_image)
        
        assert result.image.shape == sample_image.shape
        assert result.image.dtype == np.float32
    
    def test_stylize_output_range(self, config, sample_image):
        """测试输出范围"""
        stylizer = AnimeGANStylizer("Paprika", config)
        result = stylizer.stylize(sample_image)
        
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0
    
    def test_different_styles_different_output(self, config, sample_image):
        """测试不同风格产生不同输出"""
        stylizer_hayao = AnimeGANStylizer("Hayao", config)
        stylizer_shinkai = AnimeGANStylizer("Shinkai", config)
        
        result_hayao = stylizer_hayao.stylize(sample_image)
        result_shinkai = stylizer_shinkai.stylize(sample_image)
        
        # 两种风格应该产生不同的结果
        diff = np.abs(result_hayao.image - result_shinkai.image).mean()
        assert diff > 0.001  # 允许一些差异
    
    def test_invalid_style_raises(self, config):
        """测试无效风格抛出异常"""
        with pytest.raises(ValueError):
            AnimeGANStylizer("InvalidStyle", config)


# ========== Laplacian Pyramid 测试 ==========

class TestLaplacianPyramidFusion:
    """Laplacian 金字塔融合测试"""
    
    def test_init(self):
        """测试初始化"""
        fuser = LaplacianPyramidFusion(levels=4, blur_kernel=21)
        assert fuser.levels == 4
        assert fuser.blur_kernel == 21
    
    def test_fuse_output_shape(self, sample_image):
        """测试输出形状"""
        fuser = LaplacianPyramidFusion(levels=4)
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape),
            "Shinkai": create_mock_candidate("Shinkai", sample_image.shape),
            "Hayao": create_mock_candidate("Hayao", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = fuser.fuse(candidates, routing, seg_out)
        
        assert result.shape == sample_image.shape
        assert result.dtype == np.float32
    
    def test_fuse_output_range(self, sample_image):
        """测试输出范围"""
        fuser = LaplacianPyramidFusion(levels=4)
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = fuser.fuse(candidates, routing, seg_out)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_fuse_with_different_levels(self, sample_image):
        """测试不同金字塔层数"""
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        fuser_4 = LaplacianPyramidFusion(levels=4)
        fuser_6 = LaplacianPyramidFusion(levels=6)
        
        result_4 = fuser_4.fuse(candidates, routing, seg_out)
        result_6 = fuser_6.fuse(candidates, routing, seg_out)
        
        # 两种配置都应该正常工作
        assert result_4.shape == sample_image.shape
        assert result_6.shape == sample_image.shape


class TestFusionModulePhase2:
    """FusionModule Phase 2 测试"""
    
    def test_pyramid_method(self, config, sample_image):
        """测试 pyramid 融合方法"""
        module = FusionModule(config)
        
        candidates = {
            "Traditional": create_mock_candidate("Traditional", sample_image.shape)
        }
        seg_out = create_mock_seg_output(sample_image.shape)
        routing = create_mock_routing()
        
        result = module.fuse(candidates, routing, seg_out, method="laplacian_pyramid")
        
        assert result.shape == sample_image.shape
    
    def test_default_method_is_pyramid(self, config, sample_image):
        """测试默认方法是 pyramid"""
        # 配置中设置了 default_method: "laplacian_pyramid"
        module = FusionModule(config)
        assert module.default_method == "laplacian_pyramid"


# ========== 集成测试 ==========

class TestPhase2Integration:
    """Phase 2 集成测试"""
    
    def test_multi_style_pipeline(self, config, sample_image):
        """测试多风格流水线"""
        from src.stylizers.base import init_stylizers
        
        # 初始化所有风格化器
        stylizers = init_stylizers(config)
        
        # 应该包含 Traditional
        assert "Traditional" in stylizers
        
        # 应该能处理多种风格
        for style_id, stylizer in stylizers.items():
            result = stylizer.stylize(sample_image)
            assert result.image.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

