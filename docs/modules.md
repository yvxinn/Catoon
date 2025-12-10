# 模块详细规格说明

> 本文档从 design.md 拆解而来，聚焦各模块的职责、接口和实现要点。

---

## 模块总览

| 模块 | 名称 | 核心职责 | 优先级 |
|------|------|----------|--------|
| A | Preprocess | 输入标准化、Context 管理 | Phase 1 |
| B | Semantic Analysis | 语义分割 + 人脸检测 | Phase 1 |
| C | Multi-Style Candidates | 多风格候选生成 | Phase 1 |
| D | Semantic Routing | 语义到风格的映射 | Phase 1 |
| E | Region Fusion | 区域融合 | Phase 1 |
| F | Global Harmonization | 全局色彩协调 | Phase 1 |
| G | Line-art & Detail | 线稿叠加与细节 | Phase 1 |
| H | Depth Enhancement | 深度增强（可选） | Phase 3 |

---

## A. 预处理模块 (Preprocess)

### 职责
- 输入图像统一到标准格式与尺度
- 维护 Context（全局元信息、缓存、缩放回原图）

### 核心功能
- `resize_max_side`: 最长边限制为 `max_image_size`（默认 1024）
- `normalize`: 输出 float32 RGB，范围 [0,1]
- `Context` 数据结构：
  - `image_u8`, `image_f32`, `orig_size`, `proc_size`, `scale`
  - `device`（auto/cuda/cpu）
  - `cache`（候选风格图缓存、分割结果缓存）
  - `image_hash`（用于跨步骤缓存键）

### 接口
```python
class Preprocessor:
    def process(self, image_u8: np.ndarray) -> Context
    def postprocess(self, image: np.ndarray, ctx: Context) -> np.ndarray
```

### 输入/输出
- 输入：`np.ndarray uint8 (H,W,3)`
- 输出：`Context`

---

## B. 语义分析模块 (Semantic Analysis)

### 职责
- 生成可用于路由的语义 mask
- 检测人脸并生成 face mask（人脸保护）
- 可选的边界优化以降低 halo

### B1: 场景语义分割

**关键要求**：必须使用支持场景语义的分割器（SegFormer + ADE20K/SceneParse150）

**推荐方案**：
- SegFormer（mit-b0~b5）+ ADE20K 预训练
- 通过 HuggingFace Transformers 加载

**类别桶映射**（按 label 名称自动构建）：
| 桶名 | 包含类别 |
|------|----------|
| SKY | sky |
| PERSON | person |
| VEGETATION | tree, grass, plant, vegetation, flower |
| BUILDING | building, house, skyscraper, wall, window, door |
| ROAD | road, sidewalk, path, earth, ground |
| WATER | water, sea, river, lake |
| OTHERS | 其余类别 |

**接口**：
```python
class SegFormerSegmenter:
    def predict(self, image_f32: np.ndarray) -> SegmentationOutput
    
@dataclass
class SegmentationOutput:
    label_map: np.ndarray          # int32 (H,W)
    semantic_masks: dict[str, np.ndarray]  # {bucket_name: float32 mask}
    seg_logits: np.ndarray | None  # 可选，用于不确定性
```

### B2: 人脸检测

**方案**：MediaPipe Face Detector

**输出策略**：
- 多人脸：默认取 union mask
- mask 生成：bbox 扩展 20-30% → 形态学闭运算 → 可选高斯羽化

**接口**：
```python
class FaceDetector:
    def detect(self, image_u8: np.ndarray) -> np.ndarray  # float32 (H,W) mask
```

### B3: 边界优化（可选）

三种增强方案（按性价比排序）：
1. **边界带生成**：dilation/erosion 得到环形带
2. **SLIC 超像素对齐**：语义边界对齐到颜色边缘
3. **形态学 + blur**：闭运算 + Gaussian blur 生成 soft mask

---

## C. 多候选风格生成模块 (Multi-Style Candidates)

### 职责
- 对整图生成多个风格候选 `S_j`
- 缓存候选避免 UI 交互重复推理
- 缓存颜色统计用于全局协调

### C1: GAN 风格化候选

**候选示例**：
- AnimeGANv2/v3：Hayao / Shinkai / Paprika
- CartoonGAN：通用卡通风

**缓存内容**：
- `image`: float32 (H,W,3)
- `color_stats`: Lab 均值/方差

**接口**：
```python
class GANStylizer:
    def stylize(self, image: np.ndarray) -> StyleCandidate
    
@dataclass
class StyleCandidate:
    image: np.ndarray
    color_stats: ColorStats
    style_id: str
```

### C2: 传统风格化候选

**处理流程**：
```
原图 → edge-preserving 平滑 → 颜色量化（KMeans）→ toon base
```

**平滑算法选择**：
- `bilateralFilter`: 经典 cartoon 平滑（默认）
- `edgePreservingFilter`: 更"块面化"
- `pyrMeanShiftFiltering`: 海报化强

**颜色量化**：MiniBatchKMeans，默认 K=16

---

## D. 语义路由模块 (Semantic Routing)

### 职责
- 把"语义 mask + 用户配置"转换为每个区域的风格选择与参数
- 人脸保护策略对路由进行 override

### 配置结构
```yaml
region_presets:
  SKY:
    style: "Shinkai"
    mix_weight: 1.0
    toon_params: { K: 8, smooth_strength: 0.7, edge_strength: 0.3 }
  PERSON:
    style: "Hayao"
    # ...

face_policy:
  enabled: true
  mode: "protect"  # protect | blend | full_style
  face_style: "Traditional"
  gan_weight_max: 0.3
```

### 人脸保护策略
- **protect（默认）**：人脸区域强制使用 face_style，或把 GAN 权重 clamp
- **blend**：face_style 与原路由 style 混合
- **full_style**：不保护（用于对比实验）

### 接口
```python
class SemanticRouter:
    def route(
        self,
        semantic_masks: dict[str, np.ndarray],
        face_mask: np.ndarray | None,
        ui_overrides: dict
    ) -> RoutingPlan
    
@dataclass
class RoutingPlan:
    region_configs: dict[str, RegionConfig]  # {region: style/weight/params}
    face_protection_mask: np.ndarray | None
```

---

## E. 区域融合模块 (Region Fusion)

### 职责
- 将不同区域对应的候选风格图融合为一张 base 图
- 解决 halo/接缝伪影

### E1: 基础融合（默认）
```python
Base = sum(M_c' * S_{j_c}) / (sum(M_c') + eps)
```
soft mask 推荐：Gaussian blur (kernel 15~31) 或 distance transform

### E2: 多频段融合（推荐增强）
Laplacian pyramid / multiband blending
- 将候选图分解成多尺度 Laplacian 金字塔
- 在每个频段按 soft mask 融合

### E3: Poisson / seamlessClone
仅用于 boundary band 或局部 ROI（全图多区域太慢）

### 接口
```python
class FusionModule:
    def fuse(
        self,
        candidates: dict[str, StyleCandidate],
        routing: RoutingPlan,
        seg_out: SegmentationOutput,
        method: str = "soft_mask"  # soft_mask | laplacian_pyramid | poisson
    ) -> np.ndarray
```

---

## F. 全局协调模块 (Global Harmonization)

### 职责
- 解决"缝合怪"：不同区域色调不一致
- 提供全局色调滑条

### F1: 直方图匹配
使用 `skimage.exposure.match_histograms`

**参考图选择策略**：
1. `reference_region`（默认 SKY）
2. auto：面积最大区域对应的候选图
3. 用户选择的 style_id

**强度混合**（避免过于暴力）：
```python
harmonized = fused * (1 - match_strength) + matched * match_strength
```

### F2: 全局色调控制
- `gamma`: 0.5~2.0
- `contrast`: 0.5~1.5
- `saturation`: 0.5~1.5
- `brightness`: [-50, +50]
- `temperature`: 冷暖

### 接口
```python
class Harmonizer:
    def pick_reference(self, candidates, seg_out, ui_params, cfg) -> np.ndarray
    def match_and_adjust(self, fused, reference, ui_params) -> np.ndarray
```

---

## G. 线稿与细节模块 (Line-art & Detail)

### 职责
- 输出可控线稿叠加效果
- 可选的细节注入

### G1: 线稿引擎
1. **Canny**：稳定、参数易理解
2. **XDoG**：艺术线条

**叠加方式**（multiply）：
```python
output = image * (1 - edges * edge_strength)
```

### G2: 细节注入（可选）
优选：Guided Filter（opencv-contrib 的 ximgproc）

**降级方案**：若 `cv2.ximgproc` 不存在，fallback 为不开启或 bilateral 近似

### 接口
```python
class LineartEngine:
    def extract(self, image_u8: np.ndarray, params: dict) -> np.ndarray
    def overlay(self, image: np.ndarray, edges: np.ndarray, strength: float, params: dict) -> np.ndarray
```

---

## H. 深度增强模块 (Depth Enhancement) - 可选

### 职责
- 增强空气透视、纵深感

### 流程
- MiDaS 深度估计 → depth_normalized ∈ [0,1]
- 远处：降饱和、降对比、混合雾色、边缘强度衰减

### 可调参数
- `fog_strength`: 0~1
- `fog_color`: RGB
- `affect_saturation/contrast/edges`: bool

### 接口
```python
class DepthEnhancer:
    def estimate(self, image_u8: np.ndarray) -> np.ndarray
    def apply_fog(self, image: np.ndarray, depth_map: np.ndarray, params: dict) -> np.ndarray
```

---

## 模块依赖关系

```
A (Preprocess)
    ↓
B (Semantic Analysis) ←→ C (Multi-Style Candidates)
    ↓                         ↓
D (Semantic Routing) ←────────┘
    ↓
E (Region Fusion)
    ↓
F (Global Harmonization)
    ↓
G (Line-art & Detail)
    ↓
H (Depth Enhancement) [可选]
    ↓
Output
```

