# Catoon 系统架构设计文档

---

## 1. 系统概述 (System Overview)

### 1.1 设计目标

构建一个 **Training-free（无需训练）** 的语义感知图像卡通化框架。旨在解决传统全局风格迁移中“一刀切”导致的语义丢失问题（如天空被处理成草地的纹理），以及多风格拼接时产生的“缝合怪”与“边缘伪影”现象。

### 1.2 核心创新

1.  **语义级精细控制**：利用 SegFormer 语义分割与 MediaPipe 人脸检测，实现对图像不同区域（天空、建筑、人物等）的差异化风格处理。
2.  **多模态风格路由**：支持 Traditional（传统算法）、GAN（生成对抗网络）、Diffusion（扩散模型）三种引擎的混合调度。
3.  **无缝区域融合**：提出基于拉普拉斯金字塔（Laplacian Pyramid）的多频段融合算法，有效消除区域交界处的 Halo 伪影。
4.  **工程化设计**：采用模块化架构，支持组件懒加载（Lazy Loading）、并行处理与会话状态隔离，适配多用户并发场景。

---

## 2. 整体架构 (Architecture)

系统采用 **Pipeline** 设计模式，数据流通过全局 `Context` 对象在各模块间传递。

```mermaid
graph TD
    Input[输入图像] --> A[预处理 Preprocess]
    A --> Context((Context 上下文))

    subgraph SemanticAnalysis [语义分析模块]
        Context --> Seg[SegFormer 语义分割]
        Context --> Face[MediaPipe 人脸检测]
        Seg --> Buckets[语义桶映射 BucketMapper]
    end

    subgraph StyleGeneration [多风格生成模块]
        Context --> Trad[Traditional Engine]
        Context --> GAN[AnimeGAN Engine]
        Context --> Diff[Diffusion Engine]
        Trad --> RegionStyle[RegionStylizer (区域并行生成)]
        GAN --> RegionStyle
    end

    subgraph RoutingFusion [路由与融合模块]
        Buckets --> Router[语义路由器 SemanticRouter]
        Face --> Router
        Router --> Plan[路由计划 RoutingPlan]

        RegionStyle --> Fuser[区域融合器 FusionModule]
        Plan --> Fuser
        Seg --> Fuser
    end

    subgraph PostProcess [后处理模块]
        Fuser --> Harmo[全局协调 Harmonizer]
        Harmo --> Line[线稿叠加 LineartEngine]
        Line --> Output[最终输出]
    end
```

---

## 3\. 核心模块详解 (Module Details)

### A. 预处理模块 (src/preprocess)

- **职责**：标准化输入图像，维护全局上下文。
- **核心类**：`Preprocessor`
- **实现细节**：
  - **尺寸标准化**：将图像长边限制在 1024px，保持长宽比，减少显存占用。
  - **Context 管理**：创建 `Context` 数据类，封装原始图像、浮点图像、设备信息及中间结果缓存（Cache），贯穿整个生命周期。

### B. 语义分析模块 (src/segmentation)

- **职责**：理解图像内容，生成语义掩码。
- **核心类**：
  - `SegFormerSegmenter`: 加载 SegFormer 模型（mit-b2），解析 ADE20K 的 150 类标签。
  - `BucketMapper`: 将 150 个细分类别自动聚类为 7 大语义桶（SKY, PERSON, BUILDING, VEGETATION, ROAD, WATER, OTHERS）。
  - `FaceDetector`: 基于 MediaPipe 检测人脸，生成 `face_mask` 用于人脸保护策略。

### C. 风格化引擎 (src/stylizers)

- **职责**：生成不同风格的候选图像。
- **核心类**：
  - `TraditionalStylizer`: 基于双边滤波（Bilateral Filter）与 K-Means 聚类的传统卡通化算法，提供稳定的色块基准。
  - `AnimeGANStylizer`: 集成 AnimeGANv2，提供 Hayao（宫崎骏）、Shinkai（新海诚）等特定风格。
  - `DiffusionStylizer`: (Phase 3) 集成 Stable Diffusion + ControlNet，利用 Canny/Lineart 边缘约束生成高细节风格图。
  - `RegionStylizer`: **(工程亮点)** 支持多线程并行处理，仅针对有效区域生成风格图，并利用图像哈希进行缓存，避免重复推理。

### D. 语义路由模块 (src/routing)

- **职责**：决策每个像素/区域应使用什么风格。
- **核心类**：`SemanticRouter`
- **逻辑流程**：
  1.  接收语义掩码与用户 UI 配置。
  2.  查表 `region_presets` 确定每个区域的 `style_id`、`mix_weight` 和 `toon_params`。
  3.  应用 **人脸保护策略 (Face Policy)**：
      - `protect`: 强制人脸区域回退到保守风格或限制 GAN 权重。
      - `blend`: 混合风格以保持自然度。
  4.  输出 `RoutingPlan`。

### E. 区域融合模块 (src/fusion)

- **职责**：将多张风格候选图无缝拼合。
- **核心类**：`FusionModule`, `SoftMaskFusion`, `LaplacianPyramidFusion`
- **算法实现**：
  - **Soft Mask**: 使用高斯模糊处理语义掩码边缘，进行加权平均。适用于快速预览。
  - **Laplacian Pyramid (Phase 2)**: 将图像分解为拉普拉斯金字塔（默认 6 层），在不同频段分别进行 Alpha Blending，最后重建图像。有效解决了不同风格间纹理频率不一致导致的接缝问题。

### F. 全局协调模块 (src/harmonization)

- **职责**：统一画面色调，解决“缝合怪”问题。
- **核心类**：`Harmonizer`
- **实现**：采用直方图匹配（Histogram Matching）算法。
  - 自动选取“参考区域”（如面积最大的天空区域）。
  - 将融合后的图像在 Lab 色彩空间下与参考区域进行直方图对齐。
  - 支持 `strength` 参数调节匹配强度，保留原图部分色彩特征。

### G. 线稿与细节模块 (src/lineart)

- **职责**：增强画面轮廓与细节。
- **核心类**：`LineartEngine`, `CannyLineart`, `XDoGLineart`, `GuidedFilterEnhancer`
- **实现**：
  - **XDoG (Extended Difference of Gaussians)**: 生成具有手绘艺术感的线条。
  - **Guided Filter**: 使用导向滤波将原图的高频细节注入回卡通化图像，防止纹理过度平滑。

---

## 4\. 工程架构设计 (Engineering)

### 4.1 目录结构

```text
src/
├── context.py          # 数据流上下文定义
├── pipeline.py         # 主控流水线，负责模块调度
├── preprocess/         # 预处理
├── segmentation/       # 分割模型
├── stylizers/          # 风格化算法实现
├── routing/            # 路由策略
├── fusion/             # 融合算法
├── harmonization/      # 色彩协调
└── lineart/            # 线稿处理

ui/                     # UI 表现层与逻辑层分离
├── gradio_app.py       # 入口
├── layout.py           # 界面布局
├── logic.py            # 业务逻辑适配
└── state.py            # 多用户会话状态管理
```

### 4.2 关键设计模式

1.  **懒加载 (Lazy Loading)**：

    - 所有重型模型（SegFormer, AnimeGAN, Diffusion）均在 `pipeline.py` 中通过 `@property` 懒加载。
    - 优势：启动速度快，且在未启用特定功能（如未开启 Diffusion）时不占用显存。

2.  **依赖倒置与工厂模式**：

    - 风格化器继承自 `BaseStylizer`，通过 `init_stylizers` 工厂统一初始化。
    - UI 组件通过工厂函数生成，降低了代码耦合度。

3.  **多级缓存策略**：

    - **Context 级缓存**：缓存中间生成的线稿、候选图，避免同一张图重复计算。
    - **State 级缓存**（UI）：缓存上一次的渲染结果，仅在参数变更时触发局部重绘。

---

## 5\. 配置系统 (Configuration)

配置采用 YAML 格式 (`config/default.yaml`)，支持 OmegaConf 加载。

- **全局配置**：设备 (CPU/CUDA)、最大尺寸。
- **模型路径**：SegFormer、AnimeGAN、Diffusion 权重路径。
- **区域预设**：定义 SKY, PERSON 等 7 大区域的默认风格与参数。
- **策略开关**：人脸保护模式、融合算法选择等。
