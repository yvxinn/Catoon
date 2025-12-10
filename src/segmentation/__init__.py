"""
Segmentation 模块 - 语义分析

职责：
- 生成可用于路由的语义 mask (SegFormer)
- 检测人脸并生成 face mask (MediaPipe)
- 可选的边界优化
"""

from .segformer import SegFormerSegmenter
from .face import FaceDetector
from .bucket_mapper import BucketMapper, SemanticBucket

__all__ = [
    "SegFormerSegmenter",
    "FaceDetector", 
    "BucketMapper",
    "SemanticBucket"
]

