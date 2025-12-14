# 🎨 Catoon - 语义感知可控卡通化框架

> **Training-free 的语义感知可控卡通化系统**  
> 对图像不同语义区域应用不同风格，解决多风格融合的"缝合怪"和"halo 伪影"问题

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 特性

- 🗺️ **语义感知**：自动识别天空、人物、建筑、植被等区域
- 🎭 **多风格路由**：不同区域可应用不同卡通风格 (Traditional/Diffusion)
- 🔗 **无缝融合**：Soft Mask / Laplacian Pyramid 融合消除接缝伪影
- 🎨 **全局协调**：直方图匹配解决"缝合怪"问题
- 👤 **人脸保护**：防止人物面部过度风格化
- ✏️ **双线稿引擎**：Canny + XDoG 艺术线稿
- 🖥️ **交互式 UI**：模块化 Gradio 界面，实时预览与区域级调整

---

## 🏗️ 架构

```
Input Image
     │
     ▼
┌─────────────────────────────────────────────────────┐
│ A. Preprocess → B. Semantic Analysis → C. Stylizers │
│                        ↓                     ↓      │
│                 D. Semantic Routing ←────────┘      │
│                        ↓                            │
│ E. Region Fusion → F. Harmonization → G. Line-art   │
└─────────────────────────────────────────────────────┘
     │
     ▼
Output Image
```

---

## 🚀 快速开始

### 环境配置

```bash
# 创建并激活环境
conda activate catoon

# 或从头创建
conda create -n catoon python=3.10 -y
conda activate catoon

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements-lite.txt
```

### 运行 UI

```bash
cd /home/wyx/projects/Catoon
python ui/gradio_app.py
```

访问 http://localhost:7860

---

## 📁 项目结构

```
Catoon/
├── config/
│   └── default.yaml      # 配置文件
├── src/
│   ├── context.py        # 核心数据结构
│   ├── pipeline.py       # 主 Pipeline
│   ├── preprocess/       # 预处理模块
│   ├── segmentation/     # 语义分割 (SegFormer)
│   ├── stylizers/        # 风格化器 (Traditional + Diffusion)
│   ├── routing/          # 语义路由
│   ├── fusion/           # 区域融合
│   ├── harmonization/    # 全局协调
│   ├── lineart/          # 线稿生成 (Canny + XDoG)
│   └── depth/            # 深度增强 (可选)
├── ui/                   # Gradio UI (模块化)
│   ├── gradio_app.py     # 入口点
│   ├── state.py          # 会话状态管理
│   ├── config.py         # 参数数据类和常量
│   ├── components.py     # UI 组件工厂函数
│   ├── theme.py          # CSS 和主题定义
│   ├── layout.py         # 主布局和事件绑定
│   └── logic.py          # 业务逻辑
├── docs/
│   ├── design.md         # 完整设计文档
│   ├── dependencies.md   # 依赖说明
│   └── PROGRESS.md       # 开发进度
├── tests/                # 测试套件 (141 tests)
└── weights/              # 模型权重 (gitignore)
```

---

## 🎯 开发路线图

### Phase 1: MVP ✅ 完成

- [x] 项目结构与文档
- [x] 环境配置
- [x] SegFormer 语义分割
- [x] Traditional 风格化 (bilateral + KMeans)
- [x] Soft Mask 融合
- [x] 直方图匹配协调
- [x] Canny 线稿
- [x] 基础 UI

### Phase 2: 核心增强 ✅ 完成

- [x] AnimeGAN 风格化 (Hayao/Shinkai/Paprika)
- [x] 人脸保护机制
- [x] Laplacian Pyramid 融合
- [x] 区域级 UI 控制

### Phase 3: 展示加分 ✅ 完成

- [x] XDoG 艺术线稿
- [x] Guided Filter 细节注入 (含 fallback)
- [x] Diffusion 风格化 (ControlNet)
- [x] UI 模块化重构
- [ ] Poisson 边界修复 (可选)
- [ ] MiDaS 深度增强 (可选)

---

## 📖 文档

| 文档 | 描述 |
|------|------|
| [design.md](docs/design.md) | 完整架构设计 |
| [dependencies.md](docs/dependencies.md) | 依赖与安装 |
| [PROGRESS.md](docs/PROGRESS.md) | 开发进度追踪 |

---

## 🔧 配置

主要配置项 (`config/default.yaml`):

```yaml
global:
  max_image_size: 1024
  device: "auto"

segmentation:
  model: "segformer"
  backbone: "mit-b2"

fusion:
  default_method: "soft_mask"

harmonization:
  enabled: true
  reference_region: "SKY"

lineart:
  engine: "canny"
  default_strength: 0.5
```

---

## 🙏 致谢

- [SegFormer](https://github.com/NVlabs/SegFormer) - 语义分割
- [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN) - 动漫风格化
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) + [ControlNet](https://github.com/lllyasviel/ControlNet) - Diffusion 风格化
- [MediaPipe](https://mediapipe.dev/) - 人脸检测
- [Gradio](https://gradio.app/) - UI 框架

---

## 📄 License

MIT License

