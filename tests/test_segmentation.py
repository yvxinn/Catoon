"""
Segmentation 模块单元测试
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.segmentation.bucket_mapper import BucketMapper, SemanticBucket


@pytest.fixture
def config():
    """测试配置"""
    return OmegaConf.create({
        "global": {
            "max_image_size": 512,
            "device": "auto"
        },
        "segmentation": {
            "model": "segformer",
            "backbone": "mit-b2",
            "weights": "nvidia/segformer-b2-finetuned-ade-512-512",
            "input_size": 512,
            "use_face_detector": True,
            "use_boundary_refinement": False
        },
        "face_policy": {
            "enabled": True,
            "mode": "protect",
            "face_style": "Traditional",
            "gan_weight_max": 0.3,
            "bbox_expand_ratio": 0.25
        }
    })


@pytest.fixture
def sample_image():
    """创建测试图像"""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


class TestBucketMapper:
    """BucketMapper 测试类"""
    
    def test_init_with_default(self):
        """测试默认初始化"""
        mapper = BucketMapper()
        assert mapper.id2label is not None
        assert len(mapper.id2label) > 0
    
    def test_all_buckets_exist(self):
        """测试所有桶都存在"""
        mapper = BucketMapper()
        
        for bucket in SemanticBucket:
            ids = mapper.get_bucket_ids(bucket)
            assert isinstance(ids, list)
    
    def test_sky_mapping(self):
        """测试天空类别映射"""
        mapper = BucketMapper()
        
        # 在默认 ADE20K 映射中，sky 的 ID 是 2
        bucket = mapper.get_bucket(2)
        assert bucket == SemanticBucket.SKY
    
    def test_person_mapping(self):
        """测试人物类别映射"""
        mapper = BucketMapper()
        
        # 在默认 ADE20K 映射中，person 的 ID 是 12
        bucket = mapper.get_bucket(12)
        assert bucket == SemanticBucket.PERSON
    
    def test_vegetation_mapping(self):
        """测试植被类别映射"""
        mapper = BucketMapper()
        
        # tree(4), grass(9), plant(17) 都应该映射到 VEGETATION
        for class_id in [4, 9, 17]:
            bucket = mapper.get_bucket(class_id)
            assert bucket == SemanticBucket.VEGETATION
    
    def test_building_mapping(self):
        """测试建筑类别映射"""
        mapper = BucketMapper()
        
        # building(1), wall(0), window(8) 都应该映射到 BUILDING
        for class_id in [1, 0, 8]:
            bucket = mapper.get_bucket(class_id)
            assert bucket == SemanticBucket.BUILDING
    
    def test_water_mapping(self):
        """测试水体类别映射"""
        mapper = BucketMapper()
        
        # water(21), sea(26), river(60), lake(128) 都应该映射到 WATER
        for class_id in [21, 26, 60, 128]:
            bucket = mapper.get_bucket(class_id)
            assert bucket == SemanticBucket.WATER
    
    def test_road_mapping(self):
        """测试道路类别映射"""
        mapper = BucketMapper()
        
        # road(6), sidewalk(11), path(52) 都应该映射到 ROAD
        for class_id in [6, 11, 52]:
            bucket = mapper.get_bucket(class_id)
            assert bucket == SemanticBucket.ROAD
    
    def test_unknown_id_returns_others(self):
        """测试未知 ID 返回 OTHERS"""
        mapper = BucketMapper()
        
        # 使用一个不存在的 ID
        bucket = mapper.get_bucket(9999)
        assert bucket == SemanticBucket.OTHERS
    
    def test_get_bucket_ids(self):
        """测试获取桶的所有类别 ID"""
        mapper = BucketMapper()
        
        sky_ids = mapper.get_bucket_ids(SemanticBucket.SKY)
        assert 2 in sky_ids  # sky 的 ID
    
    def test_get_bucket_ids_by_string(self):
        """测试通过字符串获取桶的类别 ID"""
        mapper = BucketMapper()
        
        sky_ids = mapper.get_bucket_ids("SKY")
        assert 2 in sky_ids
    
    def test_custom_id2label(self):
        """测试自定义 id2label"""
        custom_mapping = {
            0: "custom_sky",
            1: "custom_person",
            2: "something_else"
        }
        
        mapper = BucketMapper(custom_mapping)
        
        assert mapper.get_bucket(0) == SemanticBucket.SKY
        assert mapper.get_bucket(1) == SemanticBucket.PERSON
        assert mapper.get_bucket(2) == SemanticBucket.OTHERS


class TestSemanticBucket:
    """SemanticBucket 枚举测试"""
    
    def test_all_buckets_defined(self):
        """测试所有桶都已定义"""
        expected_buckets = ["SKY", "PERSON", "VEGETATION", "BUILDING", "ROAD", "WATER", "OTHERS"]
        
        for bucket_name in expected_buckets:
            assert hasattr(SemanticBucket, bucket_name)
    
    def test_bucket_values(self):
        """测试桶的值"""
        assert SemanticBucket.SKY.value == "SKY"
        assert SemanticBucket.PERSON.value == "PERSON"


# ========== SegFormer 测试（需要模型下载，标记为慢测试）==========

@pytest.mark.slow
class TestSegFormerSegmenter:
    """SegFormer 分割器测试（需要下载模型）"""
    
    def test_init(self, config):
        """测试初始化（不加载模型）"""
        from src.segmentation import SegFormerSegmenter
        
        segmenter = SegFormerSegmenter(config)
        assert segmenter.input_size == 512
        assert segmenter._model is None  # 懒加载
    
    def test_predict(self, config, sample_image):
        """测试分割预测"""
        from src.segmentation import SegFormerSegmenter
        from src.context import SegmentationOutput
        
        segmenter = SegFormerSegmenter(config)
        image_f32 = sample_image.astype(np.float32) / 255.0
        
        output = segmenter.predict(image_f32)
        
        assert isinstance(output, SegmentationOutput)
        assert output.label_map.shape == (256, 256)
        assert output.label_map.dtype == np.int32
        
        # 检查所有桶的 mask 都存在
        for bucket in SemanticBucket:
            assert bucket.value in output.semantic_masks
            mask = output.semantic_masks[bucket.value]
            assert mask.shape == (256, 256)
            assert mask.dtype == np.float32
            assert mask.min() >= 0.0
            assert mask.max() <= 1.0
    
    def test_predict_hard(self, config, sample_image):
        """测试硬分割"""
        from src.segmentation import SegFormerSegmenter
        
        segmenter = SegFormerSegmenter(config)
        image_f32 = sample_image.astype(np.float32) / 255.0
        
        output = segmenter.predict_hard(image_f32)
        
        # 硬分割的 mask 应该是二值的
        for bucket_name, mask in output.semantic_masks.items():
            unique_values = np.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique_values)


# ========== FaceDetector 测试 ==========

class TestFaceDetector:
    """FaceDetector 测试类"""
    
    def test_init(self, config):
        """测试初始化"""
        from src.segmentation import FaceDetector
        
        detector = FaceDetector(config)
        assert detector.bbox_expand_ratio == 0.25
        assert detector._detector is None  # 懒加载
    
    def test_detect_no_face(self, config):
        """测试无人脸图像"""
        from src.segmentation import FaceDetector
        
        detector = FaceDetector(config)
        
        # 纯色图像不应该检测到人脸
        blank_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        face_mask = detector.detect(blank_image)
        
        # 可能返回 None 或空 mask
        if face_mask is not None:
            assert face_mask.sum() < 0.01 * face_mask.size  # 小于 1% 的面积
    
    def test_detect_output_shape(self, config, sample_image):
        """测试输出形状"""
        from src.segmentation import FaceDetector
        
        detector = FaceDetector(config)
        face_mask = detector.detect(sample_image)
        
        # 即使没有检测到人脸，返回 None 是合法的
        if face_mask is not None:
            assert face_mask.shape == (256, 256)
            assert face_mask.dtype == np.float32
    
    def test_detect_with_details(self, config, sample_image):
        """测试带详情的检测"""
        from src.segmentation import FaceDetector
        
        detector = FaceDetector(config)
        face_mask, details = detector.detect_with_details(sample_image)
        
        assert isinstance(details, list)
        if face_mask is not None:
            assert face_mask.shape == (256, 256)


if __name__ == "__main__":
    # 默认不运行慢测试
    pytest.main([__file__, "-v", "-m", "not slow"])

