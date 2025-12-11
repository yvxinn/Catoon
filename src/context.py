"""
Context - 全局处理上下文

贯穿整个 Pipeline 的核心数据结构。
"""

from dataclasses import dataclass, field
from typing import Any
import hashlib
import numpy as np


@dataclass
class Context:
    """全局处理上下文，贯穿整个 pipeline"""
    
    # 原始图像信息
    image_u8: np.ndarray           # uint8 (H,W,3) - 预处理后
    image_f32: np.ndarray          # float32 (H,W,3) [0,1] - 归一化后
    orig_size: tuple[int, int]     # (H_orig, W_orig) - 原始尺寸
    proc_size: tuple[int, int]     # (H_proc, W_proc) - 处理尺寸
    scale: float                   # proc_size / orig_size 的比例
    
    # 设备与缓存
    device: str = "auto"           # "auto" | "cuda" | "cpu"
    cache: dict[str, Any] = field(default_factory=dict)
    
    # 图像哈希（用于缓存键）
    image_hash: str = ""
    
    def __post_init__(self):
        """初始化后计算图像哈希"""
        if not self.image_hash and self.image_u8 is not None:
            self.image_hash = self._compute_hash(self.image_u8)
    
    @staticmethod
    def _compute_hash(image: np.ndarray) -> str:
        """计算图像哈希值用于缓存键"""
        # 使用降采样图像计算哈希，提高效率
        small = image[::16, ::16].tobytes()
        return hashlib.md5(small).hexdigest()[:12]
    
    def get_cache(self, key: str) -> Any | None:
        """获取缓存值"""
        return self.cache.get(key)
    
    def set_cache(self, key: str, value: Any) -> None:
        """设置缓存值"""
        self.cache[key] = value
    
    def make_cache_key(self, *args: str) -> str:
        """生成缓存键"""
        return f"{self.image_hash}_{'_'.join(args)}"


@dataclass
class ColorStats:
    """颜色统计信息，用于全局协调"""
    
    lab_mean: np.ndarray    # Lab 空间均值 (3,)
    lab_std: np.ndarray     # Lab 空间标准差 (3,)
    
    # 可选：更精细的直方图
    histogram: np.ndarray | None = None  # (3, 256) 每通道直方图


@dataclass
class StyleCandidate:
    """单个风格候选"""
    
    style_id: str              # 风格标识符，如 "Hayao", "Shinkai", "Traditional"
    image: np.ndarray          # float32 (H,W,3) [0,1]
    color_stats: ColorStats    # 颜色统计
    
    # 元信息
    model_type: str = ""       # "gan" | "traditional"
    model_name: str = ""       # 具体模型名称


@dataclass
class SegmentationOutput:
    """语义分割模块输出"""
    
    label_map: np.ndarray              # int32 (H,W) - 原始标签图
    semantic_masks: dict[str, np.ndarray]  # {bucket_name: float32 (H,W)} - 软掩码
    
    # 可选
    seg_logits: np.ndarray | None = None  # 分割 logits（用于不确定性）
    boundary_band_masks: dict[str, np.ndarray] | None = None  # 边界带掩码


@dataclass
class RegionConfig:
    """单个区域的风格配置"""
    
    style_id: str              # 使用的风格 ID
    strength: float = 1.0      # 风格化强度 0~1（0=原图，1=完全风格化）
    mix_weight: float = 1.0    # 混合权重（用于多风格混合）
    
    # Toon 参数（用于传统风格化）
    toon_K: int = 16           # KMeans 颜色数量
    smooth_strength: float = 0.6
    edge_strength: float = 0.5


@dataclass
class RoutingPlan:
    """语义路由输出"""
    
    region_configs: dict[str, RegionConfig]  # {bucket_name: RegionConfig}
    face_protection_mask: np.ndarray | None  # float32 (H,W) [0,1]
    face_policy_mode: str = "protect"        # "protect" | "blend" | "full_style"


@dataclass
class UIParams:
    """UI 传递的参数（可选覆盖默认配置）"""
    
    # 融合方法
    fusion_method: str = "soft_mask"  # soft_mask | laplacian_pyramid | poisson
    
    # 全局协调
    harmonization_enabled: bool = True
    harmonization_reference: str = "SKY"  # SKY | auto | style_id
    harmonization_strength: float = 0.8
    
    # 线稿
    edge_strength: float = 0.5
    line_engine: str = "canny"  # canny | xdog
    line_width: int = 1
    
    # 全局色调
    gamma: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    brightness: float = 0.0
    temperature: float = 0.0
    
    # 区域覆盖
    region_overrides: dict[str, dict] = field(default_factory=dict)
    # 例如: {"SKY": {"style": "Shinkai", "weight": 1.0}}
    
    # 人脸保护
    face_protect_enabled: bool = True
    face_protect_mode: str = "protect"
    face_gan_weight_max: float = 0.3
    
    # 深度增强
    depth_fog_enabled: bool = False
    fog_strength: float = 0.3
    fog_color: tuple[float, float, float] = (0.7, 0.75, 0.85)

