"""
Preprocess 模块单元测试
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.preprocess import Preprocessor
from src.context import Context


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {
            "max_image_size": 512,
            "device": "auto"
        }
    })


@pytest.fixture
def preprocessor(config):
    """创建预处理器"""
    return Preprocessor(config)


@pytest.fixture
def sample_image():
    """创建测试图像 (800x600)"""
    return np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """创建大图像 (2000x1500)"""
    return np.random.randint(0, 256, (1500, 2000, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """创建小图像 (200x300)"""
    return np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)


class TestPreprocessor:
    """Preprocessor 测试类"""
    
    def test_init(self, preprocessor):
        """测试初始化"""
        assert preprocessor.max_image_size == 512
        assert preprocessor.device in ["cuda", "cpu"]
    
    def test_process_returns_context(self, preprocessor, sample_image):
        """测试 process 返回 Context"""
        ctx = preprocessor.process(sample_image)
        assert isinstance(ctx, Context)
    
    def test_process_preserves_orig_size(self, preprocessor, sample_image):
        """测试保留原始尺寸"""
        ctx = preprocessor.process(sample_image)
        assert ctx.orig_size == (600, 800)
    
    def test_process_resize_large_image(self, preprocessor, large_image):
        """测试大图像被正确缩放"""
        ctx = preprocessor.process(large_image)
        
        # 原始尺寸应该保留
        assert ctx.orig_size == (1500, 2000)
        
        # 处理后的尺寸不应超过 max_image_size
        proc_h, proc_w = ctx.proc_size
        assert max(proc_h, proc_w) <= 512
        
        # scale 应该正确
        assert ctx.scale < 1.0
        assert ctx.scale == pytest.approx(512 / 2000, rel=0.01)
    
    def test_process_no_resize_small_image(self, preprocessor, small_image):
        """测试小图像不被缩放"""
        ctx = preprocessor.process(small_image)
        
        assert ctx.orig_size == (200, 300)
        assert ctx.proc_size == (200, 300)
        assert ctx.scale == 1.0
    
    def test_image_u8_format(self, preprocessor, sample_image):
        """测试 image_u8 格式正确"""
        ctx = preprocessor.process(sample_image)
        
        assert ctx.image_u8.dtype == np.uint8
        assert ctx.image_u8.ndim == 3
        assert ctx.image_u8.shape[2] == 3
    
    def test_image_f32_format(self, preprocessor, sample_image):
        """测试 image_f32 格式正确"""
        ctx = preprocessor.process(sample_image)
        
        assert ctx.image_f32.dtype == np.float32
        assert ctx.image_f32.ndim == 3
        assert ctx.image_f32.shape[2] == 3
        
        # 值范围应该在 [0, 1]
        assert ctx.image_f32.min() >= 0.0
        assert ctx.image_f32.max() <= 1.0
    
    def test_image_hash_generated(self, preprocessor, sample_image):
        """测试图像哈希被生成"""
        ctx = preprocessor.process(sample_image)
        
        assert ctx.image_hash != ""
        assert len(ctx.image_hash) == 12  # MD5 前 12 位
    
    def test_different_images_different_hash(self, preprocessor):
        """测试不同图像有不同哈希"""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        ctx1 = preprocessor.process(img1)
        ctx2 = preprocessor.process(img2)
        
        assert ctx1.image_hash != ctx2.image_hash
    
    def test_postprocess_restores_size(self, preprocessor, large_image):
        """测试后处理恢复原始尺寸"""
        ctx = preprocessor.process(large_image)
        
        # 模拟处理后的图像（使用 float32）
        processed = ctx.image_f32.copy()
        
        # 后处理
        result = preprocessor.postprocess(processed, ctx)
        
        assert result.shape == large_image.shape
        assert result.dtype == np.uint8
    
    def test_postprocess_uint8_input(self, preprocessor, sample_image):
        """测试后处理接受 uint8 输入"""
        ctx = preprocessor.process(sample_image)
        
        # 直接使用 uint8
        result = preprocessor.postprocess(ctx.image_u8, ctx)
        
        assert result.dtype == np.uint8
    
    def test_invalid_input_none(self, preprocessor):
        """测试空输入抛出异常"""
        with pytest.raises(ValueError, match="不能为空"):
            preprocessor.process(None)
    
    def test_invalid_input_wrong_shape(self, preprocessor):
        """测试错误形状抛出异常"""
        wrong_shape = np.zeros((100, 100), dtype=np.uint8)  # 2D
        
        with pytest.raises(ValueError, match="必须是"):
            preprocessor.process(wrong_shape)
    
    def test_cache_functionality(self, preprocessor, sample_image):
        """测试缓存功能"""
        ctx = preprocessor.process(sample_image)
        
        # 测试设置和获取缓存
        ctx.set_cache("test_key", "test_value")
        assert ctx.get_cache("test_key") == "test_value"
        
        # 测试不存在的键返回 None
        assert ctx.get_cache("nonexistent") is None
    
    def test_make_cache_key(self, preprocessor, sample_image):
        """测试缓存键生成"""
        ctx = preprocessor.process(sample_image)
        
        key1 = ctx.make_cache_key("style", "Hayao")
        key2 = ctx.make_cache_key("style", "Shinkai")
        
        # 同一图像不同参数应该有不同的键
        assert key1 != key2
        
        # 键应该包含图像哈希
        assert ctx.image_hash in key1


class TestDeviceResolution:
    """设备解析测试"""
    
    def test_device_auto_cuda(self):
        """测试 auto 设备解析"""
        import torch
        
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "auto"}
        })
        preprocessor = Preprocessor(config)
        
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert preprocessor.device == expected
    
    def test_device_explicit_cpu(self):
        """测试显式 CPU 设备"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"}
        })
        preprocessor = Preprocessor(config)
        
        assert preprocessor.device == "cpu"


class TestEdgeCases:
    """边界情况测试"""
    
    def test_exact_max_size(self):
        """测试图像刚好等于最大尺寸"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"}
        })
        preprocessor = Preprocessor(config)
        
        # 512x512 的图像
        img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        ctx = preprocessor.process(img)
        
        assert ctx.proc_size == (512, 512)
        assert ctx.scale == 1.0
    
    def test_very_small_image(self):
        """测试非常小的图像"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"}
        })
        preprocessor = Preprocessor(config)
        
        # 10x10 的图像
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        ctx = preprocessor.process(img)
        
        assert ctx.proc_size == (10, 10)
        assert ctx.scale == 1.0
    
    def test_aspect_ratio_preserved(self):
        """测试长宽比保持"""
        config = OmegaConf.create({
            "global": {"max_image_size": 512, "device": "cpu"}
        })
        preprocessor = Preprocessor(config)
        
        # 1000x500 的图像（2:1 比例）
        img = np.random.randint(0, 256, (500, 1000, 3), dtype=np.uint8)
        ctx = preprocessor.process(img)
        
        proc_h, proc_w = ctx.proc_size
        original_ratio = 1000 / 500  # 2.0
        processed_ratio = proc_w / proc_h
        
        assert processed_ratio == pytest.approx(original_ratio, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

