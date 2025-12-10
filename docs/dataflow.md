# 核心数据流与数据结构

> 本文档从 design.md 拆解而来，聚焦数据结构定义和数据流转。

---

## 数据流总图

```
Input Image (I)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ A. 预处理模块 Preprocess                                      │
│    输入: np.ndarray uint8 (H,W,3)                             │
│    输出: Context                                              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ B. 语义分析模块 Semantic Analysis                              │
│    输入: Context.image_f32, Context.image_u8                   │
│    输出: SegmentationOutput, face_mask                         │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ C. 多候选风格生成模块 Multi-Style Candidates                   │
│    输入: Context (含缓存 key)                                  │
│    输出: dict[str, StyleCandidate]                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ D. 语义路由模块 Semantic Routing                               │
│    输入: semantic_masks, face_mask, ui_overrides               │
│    输出: RoutingPlan                                           │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ E. 区域融合模块 Region Fusion                                  │
│    输入: candidates, RoutingPlan, SegmentationOutput           │
│    输出: fused_image (float32)                                 │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ F. 全局协调模块 Global Harmonization                           │
│    输入: fused_image, reference_image, ui_params               │
│    输出: harmonized_image                                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ G. 线稿与细节模块 Line-art & Detail                            │
│    输入: harmonized_image, Context.image_u8, params            │
│    输出: final_with_lines                                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ H. [可选] 深度增强 Depth Enhancement                           │
│    输入: image, depth_map, params                              │
│    输出: enhanced_image                                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
Output Image (O)
```

---

## 核心数据结构定义

### Context（全局上下文）

```python
from dataclasses import dataclass, field
from typing import Any
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
```

### SegmentationOutput（分割输出）

```python
@dataclass
class SegmentationOutput:
    """语义分割模块输出"""
    
    label_map: np.ndarray              # int32 (H,W) - 原始标签图
    semantic_masks: dict[str, np.ndarray]  # {bucket_name: float32 (H,W)} - 软掩码
    
    # 可选
    seg_logits: np.ndarray | None = None  # 分割 logits（用于不确定性）
    boundary_band_masks: dict[str, np.ndarray] | None = None  # 边界带掩码
```

### ColorStats（颜色统计）

```python
@dataclass
class ColorStats:
    """颜色统计信息，用于全局协调"""
    
    lab_mean: np.ndarray    # Lab 空间均值 (3,)
    lab_std: np.ndarray     # Lab 空间标准差 (3,)
    
    # 可选：更精细的直方图
    histogram: np.ndarray | None = None  # (3, 256) 每通道直方图
```

### StyleCandidate（风格候选）

```python
@dataclass
class StyleCandidate:
    """单个风格候选"""
    
    style_id: str              # 风格标识符，如 "Hayao", "Shinkai", "Traditional"
    image: np.ndarray          # float32 (H,W,3) [0,1]
    color_stats: ColorStats    # 颜色统计
    
    # 元信息
    model_type: str = ""       # "gan" | "traditional"
    model_name: str = ""       # 具体模型名称
```

### RegionConfig（区域配置）

```python
@dataclass
class RegionConfig:
    """单个区域的风格配置"""
    
    style_id: str              # 使用的风格 ID
    mix_weight: float = 1.0    # 混合权重
    
    # Toon 参数（用于传统风格化）
    toon_K: int = 16           # KMeans 颜色数量
    smooth_strength: float = 0.6
    edge_strength: float = 0.5
```

### RoutingPlan（路由计划）

```python
@dataclass
class RoutingPlan:
    """语义路由输出"""
    
    region_configs: dict[str, RegionConfig]  # {bucket_name: RegionConfig}
    face_protection_mask: np.ndarray | None  # float32 (H,W) [0,1]
    face_policy_mode: str = "protect"        # "protect" | "blend" | "full_style"
```

### UIParams（UI 参数）

```python
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
```

---

## 语义桶（Semantic Buckets）

| 桶名 | 英文标签匹配规则 | 典型类别 |
|------|-----------------|----------|
| `SKY` | `label == "sky"` | sky |
| `PERSON` | `label == "person"` | person |
| `VEGETATION` | `label in {tree, grass, plant, vegetation, flower, ...}` | 植被类 |
| `BUILDING` | `label in {building, house, skyscraper, wall, window, door, ...}` | 建筑类 |
| `ROAD` | `label in {road, sidewalk, path, earth, ground, ...}` | 道路/地面类 |
| `WATER` | `label in {water, sea, river, lake, ...}` | 水体类 |
| `OTHERS` | 其余所有类别 | 默认 |

---

## 缓存策略

### 缓存键设计

```python
def make_cache_key(image_hash: str, style_id: str, model_version: str) -> str:
    """生成风格候选的缓存键"""
    return f"{image_hash}_{style_id}_{model_version}"

def make_seg_cache_key(image_hash: str, model_name: str) -> str:
    """生成分割结果的缓存键"""
    return f"seg_{image_hash}_{model_name}"
```

### 缓存生命周期

| 缓存类型 | 存储位置 | 生命周期 | 清理时机 |
|----------|----------|----------|----------|
| 风格候选 | `Context.cache` | 单次会话 | 图像更换时 |
| 分割结果 | `Context.cache` | 单次会话 | 图像更换时 |
| 颜色统计 | `StyleCandidate.color_stats` | 随候选 | 随候选清理 |

---

## 图像格式约定

| 阶段 | 格式 | 范围 | 备注 |
|------|------|------|------|
| 输入 | `uint8` | [0, 255] | RGB 顺序 |
| 内部处理 | `float32` | [0.0, 1.0] | RGB 顺序 |
| SegFormer 输入 | `float32` | ImageNet 归一化 | 按模型要求 |
| GAN 输入 | 取决于模型 | 按模型要求 | 可能是 [-1,1] |
| 输出 | `uint8` | [0, 255] | RGB 顺序 |

---

## UI 参数映射示例

```python
# Gradio UI 传递给 Pipeline 的参数示例
ui_params = {
    "fusion_method": "soft_mask",
    "harmonization_enabled": True,
    "harmonization_reference": "SKY",
    "harmonization_strength": 0.8,
    "edge_strength": 0.5,
    "gamma": 1.0,
    "contrast": 1.1,
    "saturation": 1.0,
    "temperature": 0,
    
    "region_overrides": {
        "SKY": {"style": "Shinkai", "weight": 1.0},
        "PERSON": {"style": "Hayao", "weight": 1.0}
    },
    
    "face_protect_enabled": True,
    "face_protect_mode": "protect",
    "face_gan_weight_max": 0.3,
    
    "depth_fog_enabled": False,
    "fog_strength": 0.3
}
```

