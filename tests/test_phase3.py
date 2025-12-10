"""
Phase 3 测试 - XDoG 线稿和 Guided Filter 细节注入
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.lineart import XDoGLineart, GuidedFilterEnhancer, LineartEngine, CannyLineart


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {"max_image_size": 512, "device": "cpu"},
        "lineart": {
            "engine": "xdog",
            "default_strength": 0.5,
            "canny_low": 100,
            "canny_high": 200,
            "line_width": 1,
            "xdog_sigma": 0.5,
            "xdog_k": 1.6,
            "xdog_p": 19.0
        }
    })


@pytest.fixture
def sample_image_u8():
    """测试图像 uint8"""
    # 创建有边缘的测试图像
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:64, :64] = 255  # 左上白
    img[64:, 64:] = 128  # 右下灰
    return img


@pytest.fixture
def sample_image_f32():
    """测试图像 float32"""
    return np.random.rand(128, 128, 3).astype(np.float32)


# ========== XDoG 测试 ==========

class TestXDoGLineart:
    """XDoG 线稿测试"""
    
    def test_init(self):
        """测试初始化"""
        xdog = XDoGLineart(sigma=0.5, k=1.6, p=19.0)
        assert xdog.sigma == 0.5
        assert xdog.k == 1.6
        assert xdog.p == 19.0
    
    def test_extract_output_shape(self, sample_image_u8):
        """测试提取输出形状"""
        xdog = XDoGLineart()
        edges = xdog.extract(sample_image_u8)
        
        assert edges.shape == sample_image_u8.shape[:2]
        assert edges.dtype == np.float32
    
    def test_extract_output_range(self, sample_image_u8):
        """测试提取输出范围"""
        xdog = XDoGLineart()
        edges = xdog.extract(sample_image_u8)
        
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0
    
    def test_extract_grayscale_input(self):
        """测试灰度输入"""
        xdog = XDoGLineart()
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        
        edges = xdog.extract(gray)
        
        assert edges.shape == gray.shape
    
    def test_overlay_output_shape(self, sample_image_f32):
        """测试叠加输出形状"""
        xdog = XDoGLineart()
        edges = np.random.rand(128, 128).astype(np.float32)
        
        result = xdog.overlay(sample_image_f32, edges, strength=0.5)
        
        assert result.shape == sample_image_f32.shape
    
    def test_overlay_strength_zero(self, sample_image_f32):
        """测试零强度叠加"""
        xdog = XDoGLineart()
        edges = np.ones((128, 128), dtype=np.float32)
        
        result = xdog.overlay(sample_image_f32, edges, strength=0.0)
        
        np.testing.assert_array_almost_equal(result, sample_image_f32)
    
    def test_different_params_different_output(self, sample_image_u8):
        """测试不同参数产生不同输出"""
        xdog1 = XDoGLineart(sigma=0.3, p=10.0)
        xdog2 = XDoGLineart(sigma=0.8, p=30.0)
        
        edges1 = xdog1.extract(sample_image_u8)
        edges2 = xdog2.extract(sample_image_u8)
        
        # 不同参数应该产生不同结果
        diff = np.abs(edges1 - edges2).mean()
        assert diff > 0.001
    
    def test_line_width_effect(self, sample_image_u8):
        """测试线条宽度效果"""
        xdog_thin = XDoGLineart(line_width=1)
        xdog_thick = XDoGLineart(line_width=3)
        
        edges_thin = xdog_thin.extract(sample_image_u8)
        edges_thick = xdog_thick.extract(sample_image_u8)
        
        # 粗线条应该有更多非零像素
        assert edges_thick.sum() >= edges_thin.sum()


class TestXDoGVsCanny:
    """XDoG 与 Canny 对比测试"""
    
    def test_both_produce_edges(self, sample_image_u8):
        """测试两者都能产生边缘"""
        xdog = XDoGLineart()
        canny = CannyLineart()
        
        edges_xdog = xdog.extract(sample_image_u8)
        edges_canny = canny.extract(sample_image_u8)
        
        # 两者都应该产生边缘
        assert edges_xdog.sum() > 0
        assert edges_canny.sum() > 0
    
    def test_different_characteristics(self, sample_image_u8):
        """测试两者有不同特征"""
        xdog = XDoGLineart()
        canny = CannyLineart()
        
        edges_xdog = xdog.extract(sample_image_u8)
        edges_canny = canny.extract(sample_image_u8)
        
        # 两种方法应该产生不同的结果
        diff = np.abs(edges_xdog - edges_canny).mean()
        assert diff > 0.01  # 允许一些差异


# ========== Guided Filter 测试 ==========

class TestGuidedFilterEnhancer:
    """Guided Filter 增强器测试"""
    
    def test_init(self):
        """测试初始化"""
        gf = GuidedFilterEnhancer(radius=8, eps=0.02)
        assert gf.radius == 8
        assert gf.eps == 0.02
    
    def test_is_available_static(self):
        """测试可用性检查"""
        # 这个测试只检查方法存在
        available = GuidedFilterEnhancer.is_available()
        assert isinstance(available, bool)
    
    def test_enhance_output_shape(self, sample_image_f32):
        """测试增强输出形状"""
        gf = GuidedFilterEnhancer()
        
        result = gf.enhance(sample_image_f32)
        
        assert result.shape == sample_image_f32.shape
        assert result.dtype == np.float32
    
    def test_enhance_output_range(self, sample_image_f32):
        """测试增强输出范围"""
        gf = GuidedFilterEnhancer()
        
        result = gf.enhance(sample_image_f32)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_enhance_with_custom_guide(self, sample_image_f32):
        """测试使用自定义引导图"""
        gf = GuidedFilterEnhancer()
        guide = np.random.rand(128, 128, 3).astype(np.float32)
        
        result = gf.enhance(sample_image_f32, guide=guide)
        
        assert result.shape == sample_image_f32.shape
    
    def test_fallback_when_ximgproc_unavailable(self, sample_image_f32):
        """测试 ximgproc 不可用时的降级"""
        gf = GuidedFilterEnhancer(fallback_method="bilateral")
        
        # 即使 ximgproc 不可用，也应该能工作
        result = gf.enhance(sample_image_f32)
        
        assert result.shape == sample_image_f32.shape
    
    def test_extract_detail_layer(self, sample_image_f32):
        """测试细节层提取"""
        gf = GuidedFilterEnhancer()
        
        detail = gf.extract_detail_layer(sample_image_f32)
        
        assert detail.shape == sample_image_f32.shape
        assert detail.dtype == np.float32


# ========== LineartEngine 集成测试 ==========

class TestLineartEnginePhase3:
    """LineartEngine Phase 3 测试"""
    
    def test_xdog_engine_selection(self, config, sample_image_u8):
        """测试 XDoG 引擎选择"""
        engine = LineartEngine(config)
        
        edges = engine.extract(sample_image_u8, {"line_engine": "xdog"})
        
        assert edges.shape == sample_image_u8.shape[:2]
    
    def test_canny_engine_selection(self, config, sample_image_u8):
        """测试 Canny 引擎选择"""
        engine = LineartEngine(config)
        
        edges = engine.extract(sample_image_u8, {"line_engine": "canny"})
        
        assert edges.shape == sample_image_u8.shape[:2]
    
    def test_enhance_detail(self, config, sample_image_f32):
        """测试细节增强"""
        engine = LineartEngine(config)
        
        enhanced = engine.enhance_detail(sample_image_f32, strength=0.5)
        
        assert enhanced.shape == sample_image_f32.shape
    
    def test_enhance_detail_zero_strength(self, config, sample_image_f32):
        """测试零强度细节增强"""
        engine = LineartEngine(config)
        
        enhanced = engine.enhance_detail(sample_image_f32, strength=0.0)
        
        # 零强度应该返回原图
        np.testing.assert_array_almost_equal(enhanced, sample_image_f32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

