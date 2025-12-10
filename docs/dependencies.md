# 技术栈与依赖说明

> 本文档从 design.md 拆解而来，聚焦依赖库、版本建议和安装指南。

---

## 依赖概览

### 核心依赖（必需）

| 库 | 用途 | 所属模块 |
|----|------|----------|
| `torch` + `torchvision` | 深度学习框架 | B, C, H |
| `transformers` | SegFormer 模型加载 | B |
| `opencv-python` | 图像处理 | A, C, E, G |
| `scikit-image` | 直方图匹配、SLIC | F, B |
| `scikit-learn` | MiniBatchKMeans | C |
| `mediapipe` | 人脸检测 | B |
| `gradio` | 交互式 UI | UI |
| `pyyaml` 或 `omegaconf` | 配置管理 | 全局 |
| `numpy` | 数组操作 | 全局 |

### 可选依赖

| 库 | 用途 | 条件 |
|----|------|------|
| `opencv-contrib-python` | Guided Filter (ximgproc) | 启用细节注入时 |
| `timm` | MiDaS 深度估计 | 启用深度增强时 |

---

## 依赖清单 (requirements.txt)

### 完整版

```txt
# Core
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
numpy>=1.24.0

# Image Processing
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0  # For guided filter
scikit-image>=0.21.0
scikit-learn>=1.3.0

# Face Detection
mediapipe>=0.10.0

# UI
gradio>=4.0.0

# Config
omegaconf>=2.3.0
pyyaml>=6.0

# Optional: Depth
timm>=0.9.0  # For MiDaS
```

### 精简版 (requirements-lite.txt)

```txt
# Core (minimum for Phase 1 MVP)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
numpy>=1.24.0

# Image Processing (basic)
opencv-python>=4.8.0
scikit-image>=0.21.0
scikit-learn>=1.3.0

# Face Detection
mediapipe>=0.10.0

# UI
gradio>=4.0.0

# Config
pyyaml>=6.0
```

---

## 模块-依赖映射

```
模块 A (Preprocess):
  └── opencv-python, numpy

模块 B (Semantic Analysis):
  ├── transformers (SegFormer)
  ├── torch, torchvision
  ├── mediapipe (人脸检测)
  └── scikit-image (SLIC, 可选)

模块 C (Multi-Style Candidates):
  ├── torch (GAN 推理)
  ├── opencv-python (传统风格化)
  └── scikit-learn (MiniBatchKMeans)

模块 D (Semantic Routing):
  └── numpy, pyyaml/omegaconf

模块 E (Region Fusion):
  ├── opencv-python (soft mask, pyramid)
  └── numpy

模块 F (Global Harmonization):
  └── scikit-image (match_histograms)

模块 G (Line-art & Detail):
  ├── opencv-python (Canny)
  └── opencv-contrib-python (guided filter, 可选)

模块 H (Depth Enhancement, 可选):
  ├── torch
  └── timm (MiDaS)

UI:
  └── gradio
```

---

## 安装指南

### 方案 1：Conda 环境（推荐）

```bash
# 创建环境
conda create -n catoon python=3.10 -y
conda activate catoon

# 安装 PyTorch (CUDA 11.8 示例)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

### 方案 2：pip 直接安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 方案 3：最小安装（Phase 1 MVP）

```bash
pip install -r requirements-lite.txt
```

---

## 模型权重下载

### SegFormer（自动下载）

通过 HuggingFace Transformers 自动下载：

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
```

### AnimeGAN 权重（需手动下载）

| 模型 | 权重文件 | 下载地址 |
|------|----------|----------|
| AnimeGANv2 Hayao | `Hayao-60.onnx` | [GitHub Release](https://github.com/bryandlee/animegan2-pytorch) |
| AnimeGANv2 Shinkai | `Shinkai-53.onnx` | 同上 |
| AnimeGANv2 Paprika | `Paprika-54.onnx` | 同上 |

放置位置：`weights/animegan/`

### MiDaS 权重（可选，自动下载）

```python
import torch
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
```

---

## 潜在依赖问题与解决

### 问题 1：cv2.ximgproc 不可用

**症状**：
```python
AttributeError: module 'cv2' has no attribute 'ximgproc'
```

**原因**：`opencv-python` 不包含 contrib 模块

**解决**：
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

**降级方案**：代码中添加 fallback
```python
try:
    gf = cv2.ximgproc.guidedFilter(guide, src, radius, eps)
except AttributeError:
    gf = src  # 降级：不做细节注入
```

### 问题 2：CUDA 版本不匹配

**症状**：PyTorch 无法使用 GPU

**解决**：确认 CUDA 版本并安装匹配的 PyTorch

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题 3：MediaPipe 在某些平台安装失败

**解决**：
```bash
# 尝试指定版本
pip install mediapipe==0.10.9

# 或使用预编译 wheel
pip install --no-build-isolation mediapipe
```

---

## 版本兼容性矩阵

| Python | PyTorch | CUDA | transformers | 状态 |
|--------|---------|------|--------------|------|
| 3.10 | 2.1.x | 11.8 | 4.35.x | ✅ 推荐 |
| 3.10 | 2.0.x | 11.8 | 4.30.x | ✅ 支持 |
| 3.11 | 2.1.x | 12.1 | 4.35.x | ✅ 支持 |
| 3.9 | 1.13.x | 11.7 | 4.25.x | ⚠️ 老版本 |

---

## 推荐硬件配置

| 配置等级 | GPU | 显存 | 说明 |
|----------|-----|------|------|
| 最低 | CPU only | - | 仅支持传统风格化，速度较慢 |
| 推荐 | GTX 1660 | 6GB | 单张图像处理流畅 |
| 最佳 | RTX 3060+ | 12GB+ | 支持多候选并行、高分辨率 |

