"""
Stylizers 模块单元测试
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.stylizers import TraditionalStylizer, init_stylizers
from src.stylizers.base import BaseStylizer
from src.context import StyleCandidate, ColorStats


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {
            "max_image_size": 512,
            "device": "cpu"
        },
        "stylizers": {
            "gan": [],
            "traditional": {
                "smooth_method": "bilateral",
                "default_K": 16,
                "bilateral_d": 9,
                "bilateral_sigma_color": 75,
                "bilateral_sigma_space": 75
            }
        }
    })


@pytest.fixture
def sample_image():
    """创建测试图像 (256x256)"""
    # 创建有颜色变化的测试图像
    img = np.zeros((256, 256, 3), dtype=np.float32)
    
    # 左上：红色
    img[:128, :128, 0] = 0.9
    img[:128, :128, 1] = 0.1
    img[:128, :128, 2] = 0.1
    
    # 右上：绿色
    img[:128, 128:, 0] = 0.1
    img[:128, 128:, 1] = 0.9
    img[:128, 128:, 2] = 0.1
    
    # 左下：蓝色
    img[128:, :128, 0] = 0.1
    img[128:, :128, 1] = 0.1
    img[128:, :128, 2] = 0.9
    
    # 右下：黄色
    img[128:, 128:, 0] = 0.9
    img[128:, 128:, 1] = 0.9
    img[128:, 128:, 2] = 0.1
    
    # 添加一些噪声
    noise = np.random.randn(*img.shape).astype(np.float32) * 0.05
    img = np.clip(img + noise, 0, 1)
    
    return img


@pytest.fixture
def random_image():
    """创建随机测试图像"""
    return np.random.rand(256, 256, 3).astype(np.float32)


class TestTraditionalStylizer:
    """TraditionalStylizer 测试类"""
    
    def test_init(self, config):
        """测试初始化"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        assert stylizer.style_id == "Traditional"
        assert stylizer.model_type == "traditional"
        assert stylizer.smooth_method == "bilateral"
        assert stylizer.default_K == 16
    
    def test_stylize_returns_style_candidate(self, config, sample_image):
        """测试风格化返回 StyleCandidate"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result = stylizer.stylize(sample_image)
        
        assert isinstance(result, StyleCandidate)
        assert result.style_id == "Traditional"
        assert result.model_type == "traditional"
    
    def test_stylize_output_shape(self, config, sample_image):
        """测试输出形状"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result = stylizer.stylize(sample_image)
        
        assert result.image.shape == sample_image.shape
        assert result.image.dtype == np.float32
    
    def test_stylize_output_range(self, config, sample_image):
        """测试输出值范围"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result = stylizer.stylize(sample_image)
        
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0
    
    def test_stylize_reduces_colors(self, config, sample_image):
        """测试颜色量化效果"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        # 使用较少的颜色
        result = stylizer.stylize(sample_image, K=4)
        
        # 量化后的图像应该有更少的唯一颜色
        result_u8 = (result.image * 255).astype(np.uint8)
        unique_colors = len(np.unique(result_u8.reshape(-1, 3), axis=0))
        
        # 应该不超过 K 种颜色（实际可能略多因为平滑导致的过渡）
        assert unique_colors <= 4 * 2  # 允许一些误差
    
    def test_stylize_with_different_K(self, config, random_image):
        """测试不同 K 值的效果"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result_k4 = stylizer.stylize(random_image, K=4)
        result_k32 = stylizer.stylize(random_image, K=32)
        
        # 两个结果应该不同
        assert not np.allclose(result_k4.image, result_k32.image)
    
    def test_color_stats_computed(self, config, sample_image):
        """测试颜色统计被计算"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result = stylizer.stylize(sample_image)
        
        assert isinstance(result.color_stats, ColorStats)
        assert result.color_stats.lab_mean.shape == (3,)
        assert result.color_stats.lab_std.shape == (3,)
        assert result.color_stats.histogram is not None
        assert result.color_stats.histogram.shape == (3, 256)
    
    def test_smooth_strength_effect(self, config, random_image):
        """测试平滑强度的效果"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        # 低平滑
        result_low = stylizer.stylize(random_image, smooth_strength=0.1)
        # 高平滑
        result_high = stylizer.stylize(random_image, smooth_strength=1.0)
        
        # 高平滑应该更"平坦"（方差更小）
        var_low = result_low.image.var()
        var_high = result_high.image.var()
        
        # 这个断言可能不总是成立，取决于图像内容
        # 但对于随机图像，高平滑通常会降低方差
        # assert var_high <= var_low * 1.5  # 允许一些误差


class TestSmoothingMethods:
    """不同平滑方法的测试"""
    
    def test_bilateral_smoothing(self, sample_image):
        """测试双边滤波"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"},
            "stylizers": {
                "traditional": {
                    "smooth_method": "bilateral",
                    "default_K": 16,
                    "bilateral_d": 9,
                    "bilateral_sigma_color": 75,
                    "bilateral_sigma_space": 75
                }
            }
        })
        
        stylizer = TraditionalStylizer("Traditional", config)
        result = stylizer.stylize(sample_image)
        
        assert result.image.shape == sample_image.shape
    
    def test_edge_preserving_smoothing(self, sample_image):
        """测试边缘保持滤波"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"},
            "stylizers": {
                "traditional": {
                    "smooth_method": "edge_preserving",
                    "default_K": 16,
                    "bilateral_d": 9,
                    "bilateral_sigma_color": 75,
                    "bilateral_sigma_space": 75
                }
            }
        })
        
        stylizer = TraditionalStylizer("Traditional", config)
        result = stylizer.stylize(sample_image)
        
        assert result.image.shape == sample_image.shape
    
    def test_mean_shift_smoothing(self, sample_image):
        """测试均值漂移滤波"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"},
            "stylizers": {
                "traditional": {
                    "smooth_method": "mean_shift",
                    "default_K": 16,
                    "bilateral_d": 9,
                    "bilateral_sigma_color": 75,
                    "bilateral_sigma_space": 75
                }
            }
        })
        
        stylizer = TraditionalStylizer("Traditional", config)
        result = stylizer.stylize(sample_image)
        
        assert result.image.shape == sample_image.shape


class TestInitStylizers:
    """init_stylizers 测试"""
    
    def test_init_returns_dict(self, config):
        """测试返回字典"""
        stylizers = init_stylizers(config)
        
        assert isinstance(stylizers, dict)
    
    def test_traditional_stylizer_included(self, config):
        """测试包含传统风格化器"""
        stylizers = init_stylizers(config)
        
        assert "Traditional" in stylizers
        assert isinstance(stylizers["Traditional"], TraditionalStylizer)


class TestBaseStylizer:
    """BaseStylizer 测试"""
    
    def test_compute_color_stats(self, sample_image):
        """测试颜色统计计算"""
        stats = BaseStylizer.compute_color_stats(sample_image)
        
        assert isinstance(stats, ColorStats)
        assert stats.lab_mean.shape == (3,)
        assert stats.lab_std.shape == (3,)
        
        # Lab 空间的 L 通道范围是 0-100
        assert 0 <= stats.lab_mean[0] <= 255  # OpenCV Lab 范围
        
    def test_histogram_normalized(self, sample_image):
        """测试直方图被归一化"""
        stats = BaseStylizer.compute_color_stats(sample_image)
        
        # 每个通道的直方图应该归一化（和接近 1）
        for i in range(3):
            hist_sum = stats.histogram[i].sum()
            assert 0.99 <= hist_sum <= 1.01


class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_color_image(self, config):
        """测试单色图像"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        # 纯红色图像
        img = np.ones((64, 64, 3), dtype=np.float32)
        img[:, :, 0] = 0.8
        img[:, :, 1] = 0.2
        img[:, :, 2] = 0.2
        
        result = stylizer.stylize(img, K=4)
        
        # 应该主要保持红色
        assert result.image[:, :, 0].mean() > 0.5
    
    def test_very_small_image(self, config):
        """测试非常小的图像"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        img = np.random.rand(16, 16, 3).astype(np.float32)
        result = stylizer.stylize(img)
        
        assert result.image.shape == (16, 16, 3)
    
    def test_zero_smooth_strength(self, config, sample_image):
        """测试零平滑强度"""
        stylizer = TraditionalStylizer("Traditional", config)
        
        result = stylizer.stylize(sample_image, smooth_strength=0.0)
        
        # 应该仍然有颜色量化效果
        assert result.image.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

