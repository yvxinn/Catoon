# 语义感知可控卡通化框架

---

## 一、项目概述

### 1.1 项目目标

构建一个 **Training-free 的语义感知可控卡通化系统**，实现：

- 对图像不同语义区域（天空、人物、建筑、植被、道路、水面等）应用不同风格
- 提供丰富且可解释的可调参数（风格强度、线稿、色调、区域混合等）
- 解决多风格融合的两大典型问题：

  - “缝合怪”（跨区域色调/风格不统一）
  - “halo/接缝伪影”（mask 边界不准导致的白边/黑边/硬边）

- 提供可交互 UI（推荐 Gradio）用于实时预览与区域级编辑

### 1.2 核心创新点（答辩可用）

**多候选风格图 + 语义路由 + 区域融合 + 全局协调 + 人脸保护** 的完整 pipeline
将传统“单滤镜/单模型全图风格化”升级为**可解释、可编辑、可交互**的图像内容生成与编辑系统。

---

## 二、系统架构总览

### 2.1 核心数据流（总图）

```
Input Image (I)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ A. 预处理模块 Preprocess                                      │
│    - resize (最长边 1024，可配置)                              │
│    - RGB/float 归一化                                         │
│    - 创建 Context（缓存/设备/缩放信息）                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ B. 语义分析模块 Semantic Analysis                              │
│    B1) 场景语义分割（SegFormer + ADE20K/SceneParse150）          │
│    B2) 人脸检测（MediaPipe Face Detector）→ face mask           │
│    B3) [可选] 边界优化：SLIC 超像素 / 形态学 / 边界带生成        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ C. 多候选风格生成模块 Multi-Style Candidates                   │
│    C1) GAN 风格化：AnimeGANv2/v3（多风格），CartoonGAN 等        │
│    C2) 传统风格化：edge-preserving + KMeans 量化 + toon base     │
│    - 缓存 S_j + 颜色统计（Lab 均值方差/直方图等）                │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ D. 语义路由模块 Semantic Routing                               │
│    - region → {style_id, mix_weight, toon_params}              │
│    - face policy override（人脸保护策略覆盖）                   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ E. 区域融合模块 Region Fusion                                  │
│    E1) 默认：soft mask + 归一化加权（快）                       │
│    E2) [可选] 多频段融合：Laplacian pyramid（全图多区域更适合）  │
│    E3) [可选] Poisson/SeamlessClone：仅用于边界带/局部 ROI（慢） │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ F. 全局协调模块 Global Harmonization                           │
│    F1) 直方图匹配 match_histograms（解决“缝合怪”）              │
│    F2) 全局色调控制（gamma/contrast/saturation/temperature）     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ G. 线稿与细节模块 Line-art & Detail                            │
│    G1) 线稿引擎：Canny / XDoG                                   │
│    G2) [可选] 细节注入：Guided Filter（可降级方案）              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ H. [可选] 深度增强 Depth Enhancement                           │
│    - MiDaS 深度估计                                             │
│    - 空气透视/雾效：远处降饱和、降对比、边缘衰减、混合雾色        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
Output Image (O)
```

---

## 三、模块详细规格

## 3.1 模块 A：预处理（Preprocess）

### 职责

- 输入图像统一到标准格式与尺度
- 维护 Context（全局元信息、缓存、缩放回原图）

### 核心功能

- `resize_max_side`: 最长边限制为 `max_image_size`（默认 1024）
- `normalize`: 输出 float32 RGB，范围 [0,1]（或按模型要求进行 mean/std）
- `Context` 包含：

  - `image_u8`, `image_f32`, `orig_size`, `proc_size`, `scale`
  - `device`（auto/cuda/cpu）
  - `cache`（候选风格图缓存、分割结果缓存）
  - `image_hash`（用于跨步骤缓存键）

### 输入/输出

- 输入：`np.ndarray uint8 (H,W,3)`
- 输出：`Context`（含标准化图像与元信息）

---

## 3.2 模块 B：语义分析（Semantic Analysis）

### 职责

- 生成“可用于路由”的语义 mask（必须包含 sky/building/vegetation 等场景语义）
- 检测人脸并生成 face mask（人脸保护）
- 可选的边界优化以降低 halo

---

### B1：场景语义分割

> ⚠️ **必须使用支持场景语义的分割器**（如 SegFormer + ADE20K/SceneParse150）。
> 不建议使用 torchvision 默认的 DeepLabv3 VOC 权重（类别不含 sky/building/vegetation，会导致路由失效）。

#### 推荐方案

- SegFormer（mit-b0~b5）+ ADE20K / SceneParse150 预训练
- 通过 HuggingFace Transformers 加载（或本地权重）

#### 重要实现细节：类别映射不要“手写猜”

- 不在文档里硬编码大量类别 ID（容易错/漏/版本不一致）
- **使用官方 `id2label / label2id` 映射文件自动构建**：

  - 从模型配置 `config.id2label` 读取（Transformers 会提供）
  - 或使用公开的 ADE20K/SceneParse150 `ade20k-id2label.json` 作为真源

- 桶映射按 label 名称聚类更稳，例如：

  - SKY：label == "sky"
  - PERSON：label == "person"
  - VEGETATION：包含 tree/grass/plant/vegetation/flower 等
  - BUILDING：building/house/skyscraper/wall/window/door 等
  - ROAD：road/sidewalk/path/earth/ground 等
  - WATER：water/sea/river/lake 等
  - OTHERS：剩余类别

#### 输出形式

- `seg_logits`（可选保留，用于不确定性与后续边界带）
- `seg_label_map`：`int32 (H,W)`
- `semantic_masks`：dict[str → float32 mask (H,W)]，soft/hard 都可

---

### B2：人脸检测（Face Mask）

#### 方案

- MediaPipe Face Detector（轻量、工程友好）

#### 输出策略（工程化补强）

- 多人脸：默认取 **union mask**；可选“最大脸优先”模式（面向人像图）
- mask 生成：

  1. bbox 扩展 20–30%（覆盖额头/头发/脖子边缘）
  2. 形态学闭运算/轻微膨胀（减少碎边）
  3. optional：用高斯羽化得到 soft face mask（更自然的保护边界）

#### 输出

- `face_mask`: float32 (H,W) ∈ [0,1]

---

### B3：边界优化

目的：降低 halo/接缝伪影，提高融合边界贴合颜色边缘。

推荐三种可选增强（按性价比排序）：

1. **边界带（boundary band）生成**：
   用 dilation/erosion 得到 mask 边缘的“环形带”，用于在 E2/E3 里做重点融合
2. **SLIC 超像素对齐**：
   将语义边界在局部对齐到颜色边缘（适合复杂背景）
3. **简单形态学 + blur**：
   直接对 mask 做闭运算 + Gaussian blur，生成 soft mask

输出可附带：

- `boundary_band_masks`（每类一份或全局一份）

---

## 3.3 模块 C：多候选风格生成（Multi-Style Candidates）[重构]

变更摘要：引入基于 Diffusers 的 DiffusionStylizer 替代/补充 GAN，利用传统算法结果作为生成约束，保证候选图结构对齐、色块一致。

### 职责

- 使用 ControlNet + img2img 生成多风格候选，同时锁定结构与色块布局
- 缓存候选与颜色统计，保证 UI 交互不重复推理
- 输出对齐的候选图，便于后续语义融合无伪影

### C1：Diffusion 风格化候选（核心升级）

- **技术栈**：Stable Diffusion (SD1.5 或 SDXL-Turbo) + ControlNet (Canny/Lineart)
- **输入**
  - `init_image`: 由 C2 传统风格化生成的 toon base（提供色彩布局、简化纹理）
  - `control_map`: 由 G1 线稿引擎生成的边缘图（Canny 或 XDoG），作为 ControlNet 条件
  - `prompt`: 风格提示词（如 “Makoto Shinkai style, vibrant clouds...”）
- **控制逻辑**
  - ControlNet-Canny/Lineart 权重：0.8~1.0，用于锁死轮廓，避免人物/建筑变形，并保证多张候选图像素级对齐
  - `denoising_strength`（重绘幅度）：0.3~0.6，在传统算法生成的“色块图”上添加细节而非重绘
- **优势**
  - 传统算法控制颜色块分布，ControlNet 锁定结构，Diffusion 只负责高质量渲染，候选间天然对齐
  - 兼容现有“候选缓存 + 语义路由 + 区域融合”框架

### C2：传统风格化候选

- **职责升级**：不仅是一个可选“复古”风格，同时为 Diffusion 提供 `init_image`（色彩约束）
- **流程**：Bilateral/edge-preserving 平滑 → KMeans 量化（默认 K=16，可区域覆盖）→ toon base
- **输出**：`S_trad` 仍作为候选保留；也作为 Diffusion 的底图输入

### 接口（建议落地）

```python
class DiffusionStylizer:
    def generate_candidates(
        self,
        ctx: Context,
        traditional_image: np.ndarray,  # 来自传统算法
        edge_map: np.ndarray,           # 来自线稿模块
        styles: list[str]
    ) -> dict[str, StyleCandidate]:
        ...
```

---

## 3.4 模块 D：语义路由（Semantic Routing）

### 职责

- 把“语义 mask + 用户配置”转换为每个区域的风格选择与参数
- 人脸保护策略对路由进行 override
- 输出融合所需的 routing_map（区域 →style/权重/参数）

---

### 配置结构（YAML）

```yaml
region_presets:
  SKY:
    style: "Shinkai"
    mix_weight: 1.0
    toon_params: { K: 8, smooth_strength: 0.7, edge_strength: 0.3 }

  PERSON:
    style: "Hayao"
    mix_weight: 1.0
    toon_params: { K: 24, smooth_strength: 0.4, edge_strength: 0.6 }

  BUILDING:
    style: "CartoonGAN"
    mix_weight: 1.0
    toon_params: { K: 16, smooth_strength: 0.6, edge_strength: 0.5 }

  VEGETATION:
    style: "Paprika"
    mix_weight: 0.8
    toon_params: { K: 16, smooth_strength: 0.5, edge_strength: 0.5 }

  ROAD:
    style: "Traditional"
    mix_weight: 1.0
    toon_params: { K: 12, smooth_strength: 0.6, edge_strength: 0.4 }

  WATER:
    style: "Shinkai"
    mix_weight: 0.9
    toon_params: { K: 10, smooth_strength: 0.8, edge_strength: 0.2 }

  OTHERS:
    style: "Traditional"
    mix_weight: 1.0
    toon_params: { K: 16, smooth_strength: 0.6, edge_strength: 0.5 }

face_policy:
  enabled: true
  mode: "protect" # protect | blend | full_style
  face_style: "Traditional"
  gan_weight_max: 0.3
  preserve_skin_tone: true
```

### 人脸保护策略（落地规则建议）

- **protect（默认）**：人脸区域强制使用 `face_style`，或把 GAN 权重 clamp 到 `gan_weight_max`
- **blend**：face_style 与原路由 style 混合（例如 0.7 trad + 0.3 gan）
- **full_style**：不保护（用于对比实验）
- **区域一致性**：人脸保护阶段优先使用 PERSON 区域的“区域级候选”（含 toon_K/strength），避免回退全局 K 造成人物脸部与身体风格不一致

### 输出

- `routing_plan`: 每个 region 的 `{style_id, mix_weight, toon_params}`
- `routing_map`（可选像素级）：为融合模块提供每像素权重/风格索引（实现上可不用“每像素结构体”，用“每区域权重图”更高效）
- `face_protection_mask`：人脸保护区域

---

## 3.5 模块 E：区域融合（Region Fusion）

### 职责

- 将不同区域对应的候选风格图融合为一张 base 图
- 重点解决 halo/接缝伪影与跨区域风格差异

---

### E1：基础融合（默认，快）

- 对每个语义 mask `M_c` 生成 soft mask `M_c'`（blur/distance transform）
- 对每个类别选择对应候选 `S_{j_c}`，进行归一化加权融合：

```python
Base = sum(M_c' * S_{j_c}) / (sum(M_c') + eps)
```

soft mask 推荐：

- `Gaussian blur`（kernel 15~31）
- 或 `distance transform`（边界渐变更平滑）

---

### E2：多频段融合（推荐增强，适合“全图多区域”）

**Laplacian pyramid / multiband blending**

- 将候选图分解成多尺度 Laplacian 金字塔
- 在每个频段按 soft mask 融合
- 重建得到 Base

优势：

- 对“大范围风格差异”的边界比 simple blur 更自然
- 适合你的“多区域拼装”的核心场景

---

### E3：Poisson / seamlessClone（仅用于边界带或局部 ROI）

> 注意：Poisson/SeamlessClone 更适合“局部区域克隆”，全图多区域反复做会慢且不一定稳定。
> 推荐做法：仅对 `boundary_band` 或用户指定 ROI 做 Poisson 修复。

输出：

- `fused_image`: float32 (H,W,3)
- `blend_method`: "soft_mask" | "pyramid" | "poisson"

---

## 3.6 模块 F：全局协调（Global Harmonization）

### 职责

- 解决“缝合怪”：不同区域来自不同 stylizer 导致的色调不一致
- 提供全局色调滑条（可解释、可控）

---

### F1：直方图匹配（默认启用）

使用 `skimage.exposure.match_histograms`：

- 参考图 `reference_image` 选择策略：

  1. `reference_region`（默认 SKY 或用户指定）
  2. auto：面积最大区域对应的候选图
  3. 用户选择：某个 style_id 的候选图作为基准

```python
output = match_histograms(
    fused_image, reference_image, channel_axis=-1
)
```

> 建议保留“开关 + reference 策略”，用于消融实验对比。

问题：match_histograms 有时过于暴力，会把人脸的颜色也强制改成天空的蓝色调。
补丁：建议在直方图匹配后，再加一个 Blend (混合) 步骤。

```Python

# 伪代码
matched = match_histograms(fused, reference)
# 允许用户控制匹配强度，默认 0.8 而不是 1.0，保留一点原图的自然色
harmonized = fused * (1 - match_strength) + matched * match_strength
(你可以在你的 Config 和 UI 里加一个 harmonization_strength 参数，默认 0.8)。
```

---

### F2：全局色调控制（UI 滑条）

可调参数：

- `gamma`：0.5~2.0
- `contrast`：0.5~1.5
- `saturation`：0.5~1.5
- `brightness`：[-50, +50]
- `temperature`：冷暖（可用简单 white balance/颜色矩阵实现）

输出：

- `harmonized_image`

---

## 3.7 模块 G：线稿与细节（Line-art & Detail）[调整位置]

### 变更摘要

- **提取 (Extract)**：执行时机提前到模块 C 之前，用作 ControlNet 结构约束
- **叠加 (Overlay)**：仍保留在 Pipeline 末端，作为可选视觉强化

### 职责

- 生成边缘约束图 `edge_map`（Canny/XDoG），供 Diffusion ControlNet 使用
- 最终可选叠加线稿，增强卡通感
- 可选细节注入（Guided Filter）保留降级方案

### G1：线稿引擎（前置用于约束）

- Canny（稳定、参数易理解）
- XDoG（艺术线条，ControlNet 对应 lineart 模型更敏感）
- 输出：`edge_map float32 (H,W)`；可缓存复用

### G2：线稿叠加（末端可选）

```python
output = image * (1 - edges * edge_strength)
```

- 允许 UI 控制叠加强度、line_width、threshold
- 若 Diffusion 已显式保留线条，叠加可关闭（开关：`overlay_edges`）

### G3：细节注入（可选 + 降级）

- 优选：Guided Filter；若 `cv2.ximgproc` 不可用，自动降级为 no-op 或 bilateral 近似
- 伪代码：

```python
try:
    gf = cv2.ximgproc.guidedFilter(guide, src, radius, eps)
except Exception:
    gf = src
```

---

## 3.8 模块 H：深度增强（可选加分插件）

### 职责

- 增强空气透视、纵深感与“电影感”

流程：

- MiDaS 深度估计 → depth_normalized ∈ [0,1]（0 近 1 远）
- 远处：

  - 降饱和
  - 降对比
  - 混合雾色
  - 边缘强度衰减（让远处更柔）

可调参数：

- `fog_strength`：0~1
- `fog_color`：RGB
- `affect_saturation/contrast/edges`：bool

---

## 四、可复现实验设置（建议写入报告）

### 4.1 默认推理设置

- 输入 resize：最长边 1024
- SegFormer 输入：512（resize 或 sliding window，先做简单 resize）
- 输出：回到原图尺寸

### 4.2 默认候选集合（最小可用）

- Diffusion：Shinkai、Hayao（2 个，ControlNet-Canny）
- 传统：Traditional（1 个，亦作为 Diffusion init）

> 先保证 “2 个 Diffusion + Traditional + 路由 + 融合 + 协调 + 线稿” 闭环跑通，再扩展更多风格。

### 4.3 默认融合策略

- 默认：soft mask
- 增强：pyramid（用于展示“更自然边界”）
- Poisson：仅 boundary band（展示去 halo 修复）

---

## 五、配置系统设计

### 5.1 主配置文件结构（参考）

```yaml
global:
  max_image_size: 1024
  device: "auto"
  cache_enabled: true

segmentation:
  model: "segformer" # segformer | others
  backbone: "mit-b2"
  weights: "nvidia/segformer-b2-finetuned-ade-512-512"
  use_face_detector: true
  use_boundary_refinement: false

stylizers:
  gan:
    - { name: "Hayao", model: "AnimeGANv2", weights: "Hayao" }
    - { name: "Shinkai", model: "AnimeGANv2", weights: "Shinkai" }
    - { name: "Paprika", model: "AnimeGANv2", weights: "Paprika" }
    - { name: "CartoonGAN", model: "CartoonGAN", weights: "default" }
  traditional:
    smooth_method: "bilateral" # bilateral | edge_preserving | mean_shift
    default_K: 16

fusion:
  default_method: "soft_mask" # soft_mask | laplacian_pyramid | poisson
  soft_mask_blur_kernel: 21
  enable_seamless_blend: false
  poisson_roi: "boundary_band" # boundary_band | manual

harmonization:
  enabled: true
  reference_region: "SKY" # SKY | auto | style_id
  histogram_matching: true

lineart:
  engine: "canny" # canny | xdog
  default_strength: 0.5
  canny_low: 100
  canny_high: 200
  line_width: 1

depth:
  enabled: false
  model: "MiDaS_small"
  fog_strength: 0.3
  fog_color: [0.7, 0.75, 0.85]
  affect_edges: true

region_presets: ... # 见 D 节
face_policy: ... # 见 D 节
```

---

## 六、技术栈与依赖

### 6.1 核心依赖（建议锁版本范围）

- torch / torchvision
- transformers（SegFormer）
- opencv-python + **opencv-contrib-python**（若用 guided filter）
- scikit-image（match_histograms、slic）
- scikit-learn（MiniBatchKMeans）
- mediapipe（人脸）
- gradio（UI）
- pyyaml 或 omegaconf（配置）
- diffusers（Stable Diffusion / ControlNet）
- accelerate（Diffusers 推理调度）
- controlnet_aux（可选，若直接复用本地 Canny/XDoG 可不装）

> 硬件建议：SD1.5 + ControlNet 推荐显存 ≥8GB；若使用 SDXL-Turbo/LCM 可将显存需求降到 4–6GB。

> 工程建议：提供 `requirements.txt` + `requirements-lite.txt`（不含可选深度/poisson/guided 等重依赖）。

---

## 七、目录结构（推荐）

```
Catoon/
├── config/
│   ├── default.yaml
│   └── presets/
├── src/
│   ├── pipeline.py
│   ├── context.py
│   ├── preprocess/
│   ├── segmentation/
│   ├── stylizers/
│   ├── routing/
│   ├── fusion/
│   ├── harmonization/
│   ├── lineart/
│   ├── depth/
│   └── utils/
├── ui/                    # 模块化 Gradio UI
│   ├── gradio_app.py      # 入口点
│   ├── state.py           # 会话状态管理
│   ├── config.py          # 参数数据类和常量
│   ├── components.py      # UI 组件工厂函数
│   ├── theme.py           # CSS 和主题定义
│   ├── layout.py          # 主布局和事件绑定
│   └── logic.py           # 业务逻辑
├── tests/                 # 测试套件 (141 tests)
├── weights/               # gitignored
├── examples/
├── requirements.txt
└── README.md
```

---

## 八、主 Pipeline 伪代码（结构一致性扩散版）

```python
class CartoonPipeline:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.pre = Preprocessor(self.cfg)
        self.seg = SegFormerSegmenter(self.cfg)
        self.face = FaceDetector(self.cfg) if self.cfg.segmentation.use_face_detector else None

        self.stylizers = init_stylizers(self.cfg)           # 含 Traditional
        self.diffusion = DiffusionStylizer(self.cfg)        # 新增
        self.router = SemanticRouter(self.cfg)

        self.fuser = FusionModule(self.cfg)
        self.harmo = Harmonizer(self.cfg)
        self.line = LineartEngine(self.cfg)
        self.depth = DepthEnhancer(self.cfg) if self.cfg.depth.enabled else None

    def process(self, image_u8, ui_params=None):
        ui_params = ui_params or {}

        # A. 预处理
        ctx = self.pre.process(image_u8)

        # B. 语义分割 + 人脸
        seg_out = self.seg.predict(ctx.image_f32)
        face_mask = self.face.detect(ctx.image_u8) if self.face else None

        # [NEW] 约束生成（结构 + 色彩）
        edge_map = self.line.extract(ctx.image_u8, ui_params)           # 结构约束
        trad_candidate = self.stylizers["Traditional"].stylize(
            ctx.image_f32,
            K=ui_params.get("traditional_k", self.cfg.stylizers.traditional.default_K)
        )                                                               # 色彩约束

        # C. 候选生成（Diffusion 接管，多风格）
        candidates = self.diffusion.generate_candidates(
            ctx=ctx,
            traditional_image=trad_candidate.image,
            edge_map=edge_map,
            styles=[s.name for s in self.cfg.diffusion.styles]
        )
        candidates["Traditional"] = trad_candidate

        # D. 语义路由（可复用原有逻辑）
        routing = self.router.route(
            semantic_masks=seg_out.semantic_masks,
            face_mask=face_mask,
            ui_overrides=ui_params
        )

        # E. 区域融合
        fused = self.fuser.fuse(
            candidates=candidates,
            routing=routing,
            seg_out=seg_out,
            method=ui_params.get("fusion_method", self.cfg.fusion.default_method)
        )

        # F. 全局协调
        if ui_params.get("harmonization_enabled", self.cfg.harmonization.enabled):
            ref = self.harmo.pick_reference(candidates, seg_out, ui_params, self.cfg.harmonization)
            fused = self.harmo.match_and_adjust(fused, ref, ui_params)

        # G. 线稿叠加（可选，复用前面的 edge_map）
        if ui_params.get("overlay_edges", True):
            edge_strength = ui_params.get("edge_strength", self.cfg.lineart.default_strength)
            if edge_strength > 1e-3:
                fused = self.line.overlay(fused, edge_map, edge_strength, ui_params)

        # H. 深度增强（可选）
        if self.depth and ui_params.get("depth_fog_enabled", False):
            depth_map = self.depth.estimate(ctx.image_u8)
            fused = self.depth.apply_fog(fused, depth_map, ui_params)

        return self.pre.postprocess(fused, ctx)
```

---

## 九、UI 界面设计（Gradio）

### 9.1 UI 分层

1. **全局控制**

- style_strength（可选）
- edge_strength / line_width / line_engine
- gamma/contrast/saturation/temperature
- harmonization_enabled + reference picker
- fusion_method（soft/pyramid/poisson）
- depth_fog_enabled（可选）

2. **区域控制（折叠面板）**

- 每个 region：style 下拉 + mix_weight +（可选）K_multiplier

3. **人脸保护**

- enabled、mode、gan_weight_max、preserve_skin_tone

### 9.2 UI 参数映射（标准）

```python
ui_params = {
  "fusion_method": "soft_mask",
  "harmonization_enabled": True,
  "harmonization_reference": "SKY",  # or "auto" or style_id
  "edge_strength": 0.5,
  "gamma": 1.0, "contrast": 1.1, "saturation": 1.0, "temperature": 0,

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

---

## 十、实现优先级与里程碑

### Phase 1：MVP ✅ 完成

- [x] SegFormer 分割 + bucket 映射（按 id2label 自动构建）
- [x] Traditional stylizer（bilateral + MiniBatchKMeans）
- [x] 路由 + soft_mask 融合
- [x] match_histograms 全局协调
- [x] Canny 线稿
- [x] 简易 Gradio UI（上传 → 输出、几个滑条）

### Phase 2：核心增强 ✅ 完成

- [x] AnimeGAN（Hayao/Shinkai/Paprika）接入候选
- [x] MediaPipe 人脸检测 + face protect
- [x] Laplacian Pyramid 融合
- [x] UI：区域级风格选择 + 权重

### Phase 3：展示/答辩加分 ✅ 完成

- [x] XDoG 艺术线稿
- [x] Guided Filter 细节注入（含 fallback）
- [x] Diffusion 风格化（ControlNet + img2img）
- [x] UI 模块化重构
- [ ] SLIC/边界带 Poisson 修复（可选）
- [ ] MiDaS depth fog（可选）

---

## 十一、测试与验证

### 11.1 单元测试

- 分割输出 shape 与类别桶覆盖
- 候选缓存命中率（重复调用不重复推理）
- 融合输出数值范围与无 NaN
- harmonization 开关对比

### 11.2 消融实验（报告建议必做）

- 有/无语义路由
- 有/无全局协调（缝合怪对比）
- 有/无人脸保护（人像崩坏对比）
- soft vs pyramid（halo/边界自然度对比）

---

## 十二、潜在风险与应对（更新版）

| 风险                            | 影响 | 应对策略                                                    |
| ------------------------------- | ---- | ----------------------------------------------------------- |
| ADE20K 映射错误（手写 ID 误差） | 高   | 使用 `id2label/label2id` 自动构建桶映射（不要硬编码长列表） |
| OpenCV guidedFilter 不可用      | 中   | 强制说明依赖 `opencv-contrib-python` + fallback 降级        |
| Poisson 全图多区域太慢/不稳定   | 中   | 仅对 boundary band / ROI 使用；全图用 pyramid               |
| GAN 权重/代码难集成             | 中   | MVP 先用 Traditional + 2 个风格候选；GAN 作为 Phase2        |
| 推理慢                          | 中   | 缓存候选、SegFormer 用较小 backbone、UI 调参不重复跑 GAN    |

---

## 十三、报告/答辩关键表述（可直接复制）

### 一句话总结

> **训练-free 的语义感知可控卡通化框架**：通过“多候选风格图 + 语义路由 + 区域级融合 + 全局色彩协调 + 人脸保护”，实现可解释、可编辑、可交互的图像卡通化系统，并显著缓解多风格拼接带来的缝合怪与 halo 伪影。

### 创新点列表

1. **语义路由**：不同语义区域应用不同风格与参数
2. **多档融合策略**：soft（快）→ pyramid（自然）→ Poisson（边界修复）
3. **全局色彩协调**：直方图匹配统一色调，解决缝合怪
4. **人脸保护机制**：限制 GAN 权重/保守风格保障人像可用性
5. **可控交互 UI**：区域级风格选择 + 滑条控制构成 human-in-the-loop

---
