"""
集成测试 - 测试完整 Pipeline 流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image

from src.pipeline import CartoonPipeline, load_pipeline


@pytest.fixture
def sample_image():
    """创建测试图像 (256x256)"""
    # 创建有颜色变化的测试图像
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 上半部分：蓝色天空
    img[:100, :, 0] = 135
    img[:100, :, 1] = 206
    img[:100, :, 2] = 235
    
    # 中间：绿色植被
    img[100:180, :, 0] = 34
    img[100:180, :, 1] = 139
    img[100:180, :, 2] = 34
    
    # 下部：灰色道路
    img[180:, :, 0] = 128
    img[180:, :, 1] = 128
    img[180:, :, 2] = 128
    
    # 添加一些随机纹理
    noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


@pytest.fixture
def pipeline():
    """创建 Pipeline"""
    return load_pipeline()


class TestPipelineIntegration:
    """Pipeline 集成测试"""
    
    def test_load_pipeline(self):
        """测试加载 Pipeline"""
        pipe = load_pipeline()
        assert isinstance(pipe, CartoonPipeline)
    
    def test_process_basic(self, pipeline, sample_image):
        """测试基本处理流程"""
        result = pipeline.process(sample_image)
        
        # 检查输出格式
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_process_with_no_harmonization(self, pipeline, sample_image):
        """测试禁用协调"""
        result = pipeline.process(
            sample_image,
            ui_params={"harmonization_enabled": False}
        )
        
        assert result.shape == sample_image.shape
    
    def test_process_with_no_lineart(self, pipeline, sample_image):
        """测试禁用线稿"""
        result = pipeline.process(
            sample_image,
            ui_params={"edge_strength": 0.0}
        )
        
        assert result.shape == sample_image.shape
    
    def test_process_with_custom_params(self, pipeline, sample_image):
        """测试自定义参数"""
        ui_params = {
            "fusion_method": "soft_mask",
            "harmonization_enabled": True,
            "harmonization_strength": 0.5,
            "edge_strength": 0.3,
            "gamma": 1.2,
            "contrast": 1.1,
            "saturation": 1.0
        }
        
        result = pipeline.process(sample_image, ui_params)
        
        assert result.shape == sample_image.shape
    
    def test_process_with_region_overrides(self, pipeline, sample_image):
        """测试区域覆盖"""
        ui_params = {
            "region_overrides": {
                "SKY": {"style": "Traditional", "weight": 0.8},
                "VEGETATION": {"style": "Traditional", "weight": 0.9}
            }
        }
        
        result = pipeline.process(sample_image, ui_params)
        
        assert result.shape == sample_image.shape
    
    def test_output_differs_from_input(self, pipeline, sample_image):
        """测试输出与输入不同（实际进行了处理）"""
        result = pipeline.process(sample_image)
        
        # 输出应该与输入有所不同
        diff = np.abs(result.astype(float) - sample_image.astype(float)).mean()
        assert diff > 1.0  # 平均差异应该大于 1


class TestPipelineEdgeCases:
    """边界情况测试"""
    
    def test_small_image(self, pipeline):
        """测试小图像"""
        small_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        
        result = pipeline.process(small_img)
        
        assert result.shape == small_img.shape
    
    def test_large_image(self, pipeline):
        """测试大图像（会被缩放）"""
        large_img = np.random.randint(0, 256, (1200, 1600, 3), dtype=np.uint8)
        
        result = pipeline.process(large_img)
        
        # 应该恢复到原始尺寸
        assert result.shape == large_img.shape


class TestModuleLazyLoading:
    """模块懒加载测试"""
    
    def test_modules_not_loaded_initially(self):
        """测试模块初始时未加载"""
        pipe = load_pipeline()
        
        # 私有属性应该为 None
        assert pipe._preprocessor is None
        assert pipe._segmenter is None
        assert pipe._stylizers is None
    
    def test_modules_loaded_on_access(self):
        """测试访问时加载模块"""
        pipe = load_pipeline()
        
        # 访问 preprocessor
        _ = pipe.preprocessor
        assert pipe._preprocessor is not None
        
        # 访问 stylizers
        _ = pipe.stylizers
        assert pipe._stylizers is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

