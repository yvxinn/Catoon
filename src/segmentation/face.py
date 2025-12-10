"""
FaceDetector - 基于 MediaPipe 的人脸检测

检测图像中的人脸并生成 face mask，用于人脸保护策略。
"""

import cv2
import numpy as np
from omegaconf import DictConfig


class FaceDetector:
    """基于 MediaPipe 的人脸检测器"""
    
    def __init__(self, cfg: DictConfig):
        """
        初始化人脸检测器
        
        Args:
            cfg: 配置对象
        """
        self.cfg = cfg
        
        # 从 face_policy 获取配置
        face_cfg = cfg.face_policy
        self.bbox_expand_ratio = face_cfg.get("bbox_expand_ratio", 0.25)
        
        # 延迟加载检测器
        self._detector = None
    
    @property
    def detector(self):
        """懒加载 MediaPipe 检测器"""
        if self._detector is None:
            self._load_detector()
        return self._detector
    
    def _load_detector(self) -> None:
        """加载 MediaPipe Face Detection"""
        import mediapipe as mp
        
        print("[FaceDetector] Loading MediaPipe Face Detection...")
        
        mp_face_detection = mp.solutions.face_detection
        self._detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = 全范围模型，适合远距离人脸
            min_detection_confidence=0.5
        )
        
        print("[FaceDetector] Loaded successfully")
    
    def detect(self, image_u8: np.ndarray) -> np.ndarray | None:
        """
        检测人脸并生成 mask
        
        Args:
            image_u8: 输入图像，uint8 (H,W,3) RGB
        
        Returns:
            人脸 mask，float32 (H,W) [0,1]，如果没有检测到人脸返回 None
        """
        h, w = image_u8.shape[:2]
        
        # MediaPipe 需要 RGB 输入
        results = self.detector.process(image_u8)
        
        if not results.detections:
            return None
        
        # 创建空 mask
        face_mask = np.zeros((h, w), dtype=np.float32)
        
        # 处理每个检测到的人脸
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # 转换相对坐标到绝对坐标
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # 扩展边界框
            expand_w = int(box_w * self.bbox_expand_ratio)
            expand_h = int(box_h * self.bbox_expand_ratio)
            
            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(w, x + box_w + expand_w)
            y2 = min(h, y + box_h + expand_h)
            
            # 在 mask 上绘制矩形（硬边界）
            face_mask[y1:y2, x1:x2] = 1.0
        
        # 应用形态学闭运算，填充空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        
        # 应用高斯模糊生成 soft mask
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
        
        return face_mask
    
    def detect_with_details(
        self, 
        image_u8: np.ndarray
    ) -> tuple[np.ndarray | None, list[dict]]:
        """
        检测人脸并返回详细信息
        
        Args:
            image_u8: 输入图像，uint8 (H,W,3) RGB
        
        Returns:
            (face_mask, face_details) 元组
            face_details 是检测到的人脸信息列表
        """
        h, w = image_u8.shape[:2]
        
        results = self.detector.process(image_u8)
        
        if not results.detections:
            return None, []
        
        face_mask = np.zeros((h, w), dtype=np.float32)
        face_details = []
        
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # 转换坐标
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)
            
            # 扩展边界框
            expand_w = int(box_w * self.bbox_expand_ratio)
            expand_h = int(box_h * self.bbox_expand_ratio)
            
            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(w, x + box_w + expand_w)
            y2 = min(h, y + box_h + expand_h)
            
            face_mask[y1:y2, x1:x2] = 1.0
            
            face_details.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": detection.score[0] if detection.score else 0.0,
                "area": (x2 - x1) * (y2 - y1)
            })
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
        
        return face_mask, face_details
    
    def get_largest_face_mask(self, image_u8: np.ndarray) -> np.ndarray | None:
        """
        只获取最大人脸的 mask
        
        Args:
            image_u8: 输入图像
        
        Returns:
            最大人脸的 mask
        """
        face_mask, face_details = self.detect_with_details(image_u8)
        
        if face_mask is None or not face_details:
            return None
        
        # 找到面积最大的人脸
        largest = max(face_details, key=lambda x: x["area"])
        
        # 创建只包含最大人脸的 mask
        h, w = image_u8.shape[:2]
        single_mask = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = largest["bbox"]
        single_mask[y1:y2, x1:x2] = 1.0
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        single_mask = cv2.morphologyEx(single_mask, cv2.MORPH_CLOSE, kernel)
        single_mask = cv2.GaussianBlur(single_mask, (21, 21), 0)
        
        return single_mask
    
    def visualize(
        self, 
        image_u8: np.ndarray,
        face_mask: np.ndarray | None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        可视化人脸检测结果
        
        Args:
            image_u8: 原始图像
            face_mask: 人脸 mask
            alpha: 叠加透明度
        
        Returns:
            可视化图像
        """
        if face_mask is None:
            return image_u8.copy()
        
        # 创建红色叠加层
        h, w = image_u8.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 2] = (face_mask * 255).astype(np.uint8)  # 红色通道
        
        # 叠加
        result = cv2.addWeighted(image_u8, 1 - alpha, overlay, alpha, 0)
        
        return result


def create_face_detector(cfg: DictConfig) -> FaceDetector:
    """
    便捷函数：创建人脸检测器
    
    Args:
        cfg: 配置对象
    
    Returns:
        FaceDetector 实例
    """
    return FaceDetector(cfg)

