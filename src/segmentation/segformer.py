"""
SegFormerSegmenter - 基于 SegFormer 的语义分割

使用 HuggingFace Transformers 加载预训练的 SegFormer 模型，
对图像进行语义分割并生成语义桶的 soft mask。
"""

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

from ..context import SegmentationOutput
from .bucket_mapper import BucketMapper, SemanticBucket


class SegFormerSegmenter:
    """基于 SegFormer 的语义分割器"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化分割器
        
        Args:
            cfg: 配置对象，需包含 segmentation 相关配置
        """
        self.cfg = cfg.segmentation
        self.input_size = self.cfg.input_size  # 默认 512
        
        # 解析设备
        global_cfg = getattr(cfg, 'global')
        device_str = global_cfg.device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        # 延迟加载模型
        self._model = None
        self._processor = None
        self._bucket_mapper = None
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def processor(self):
        """懒加载处理器"""
        if self._processor is None:
            self._load_model()
        return self._processor
    
    @property
    def bucket_mapper(self) -> BucketMapper:
        """懒加载桶映射器"""
        if self._bucket_mapper is None:
            self._load_model()
        return self._bucket_mapper
    
    def _load_model(self) -> None:
        """加载 SegFormer 模型和处理器"""
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        
        model_name = self.cfg.weights  # e.g., "nvidia/segformer-b2-finetuned-ade-512-512"
        
        print(f"[SegFormer] Loading model: {model_name}")
        
        # 加载模型和处理器（使用 safetensors 格式避免 PyTorch 版本问题）
        self._processor = SegformerImageProcessor.from_pretrained(model_name)
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True  # 使用 safetensors 格式
        )
        self._model.to(self.device)
        self._model.eval()
        
        # 从模型配置获取 id2label 并创建桶映射
        id2label = self._model.config.id2label
        # 转换键为 int（有时是字符串）
        id2label = {int(k): v for k, v in id2label.items()}
        self._bucket_mapper = BucketMapper(id2label)
        
        print(f"[SegFormer] Model loaded on {self.device}")
        print(f"[SegFormer] Number of classes: {len(id2label)}")
    
    def predict(self, image_f32: np.ndarray) -> SegmentationOutput:
        """
        对图像进行语义分割
        
        Args:
            image_f32: 输入图像，float32 (H,W,3) [0,1] RGB
        
        Returns:
            SegmentationOutput 包含 label_map 和 semantic_masks
        """
        # 转换为 uint8 用于处理器
        image_u8 = (image_f32 * 255).astype(np.uint8)
        orig_h, orig_w = image_u8.shape[:2]
        
        # 使用处理器预处理
        inputs = self.processor(images=image_u8, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes, H', W')
        
        # 上采样到原始尺寸
        logits_upsampled = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        # 获取预测标签
        label_map = logits_upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        
        # 获取 softmax 概率用于 soft mask
        probs = torch.softmax(logits_upsampled, dim=1).squeeze(0)  # (num_classes, H, W)
        probs_np = probs.cpu().numpy()
        
        # 生成每个语义桶的 soft mask
        semantic_masks = self._generate_semantic_masks(label_map, probs_np)
        
        # 保存 logits 用于不确定性分析（可选）
        seg_logits = logits_upsampled.squeeze(0).cpu().numpy()
        
        return SegmentationOutput(
            label_map=label_map,
            semantic_masks=semantic_masks,
            seg_logits=seg_logits
        )
    
    def _generate_semantic_masks(
        self, 
        label_map: np.ndarray,
        probs: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        生成语义桶的 soft mask
        
        Args:
            label_map: 预测的标签图 (H, W)
            probs: softmax 概率 (num_classes, H, W)
        
        Returns:
            {bucket_name: soft_mask} 字典
        """
        semantic_masks = {}
        
        for bucket in SemanticBucket:
            # 获取该桶包含的所有类别 ID
            class_ids = self.bucket_mapper.get_bucket_ids(bucket)
            
            if not class_ids:
                # 如果该桶没有类别，创建全零 mask
                mask = np.zeros(label_map.shape, dtype=np.float32)
            else:
                # 聚合该桶所有类别的概率
                mask = np.zeros(label_map.shape, dtype=np.float32)
                for class_id in class_ids:
                    if class_id < probs.shape[0]:
                        mask += probs[class_id]
                
                # Clamp to [0, 1]
                mask = np.clip(mask, 0, 1)
            
            semantic_masks[bucket.value] = mask
        
        return semantic_masks
    
    def predict_hard(self, image_f32: np.ndarray) -> SegmentationOutput:
        """
        生成硬分割（binary mask）
        
        Args:
            image_f32: 输入图像，float32 (H,W,3) [0,1] RGB
        
        Returns:
            SegmentationOutput，但 semantic_masks 是 binary 的
        """
        output = self.predict(image_f32)
        
        # 将 soft mask 转换为 hard mask
        hard_masks = {}
        for bucket_name, soft_mask in output.semantic_masks.items():
            # 使用 label_map 生成精确的 binary mask
            class_ids = self.bucket_mapper.get_bucket_ids(bucket_name)
            hard_mask = np.isin(output.label_map, class_ids).astype(np.float32)
            hard_masks[bucket_name] = hard_mask
        
        return SegmentationOutput(
            label_map=output.label_map,
            semantic_masks=hard_masks,
            seg_logits=output.seg_logits
        )
    
    def visualize(
        self, 
        image_f32: np.ndarray, 
        seg_output: SegmentationOutput,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        可视化分割结果
        
        Args:
            image_f32: 原始图像
            seg_output: 分割输出
            alpha: 叠加透明度
        
        Returns:
            可视化图像 uint8
        """
        # 语义桶的颜色映射
        bucket_colors = {
            SemanticBucket.SKY.value: [135, 206, 235],      # 天蓝色
            SemanticBucket.PERSON.value: [255, 192, 203],   # 粉色
            SemanticBucket.VEGETATION.value: [34, 139, 34], # 森林绿
            SemanticBucket.BUILDING.value: [128, 128, 128], # 灰色
            SemanticBucket.ROAD.value: [70, 70, 70],        # 深灰
            SemanticBucket.WATER.value: [0, 0, 255],        # 蓝色
            SemanticBucket.OTHERS.value: [255, 255, 255],   # 白色
        }
        
        # 创建彩色分割图
        h, w = seg_output.label_map.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        
        for bucket_name, mask in seg_output.semantic_masks.items():
            color = bucket_colors.get(bucket_name, [255, 255, 255])
            # 使用 soft mask 的阈值
            binary_mask = mask > 0.5
            color_seg[binary_mask] = color
        
        # 叠加到原图
        image_u8 = (image_f32 * 255).astype(np.uint8)
        overlay = cv2.addWeighted(image_u8, 1 - alpha, color_seg, alpha, 0)
        
        return overlay


def create_segmenter(cfg: DictConfig) -> SegFormerSegmenter:
    """
    便捷函数：创建分割器
    
    Args:
        cfg: 配置对象
    
    Returns:
        SegFormerSegmenter 实例
    """
    return SegFormerSegmenter(cfg)

