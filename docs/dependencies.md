# 技术栈与依赖说明

---

## 依赖概览

### 核心依赖

| 库                      | 用途               | 所属模块   |
| ----------------------- | ------------------ | ---------- |
| `torch` + `torchvision` | 深度学习框架       | B, C, H    |
| `transformers`          | SegFormer 模型加载 | B          |
| `opencv-python`         | 图像处理           | A, C, E, G |
| `scikit-image`          | 直方图匹配         | F          |
| `scikit-learn`          | MiniBatchKMeans    | C          |
| `mediapipe`             | 人脸检测           | B          |
| `gradio`                | 交互式 UI          | UI         |
| `omegaconf`             | 配置管理           | 全局       |

### 可选依赖

| 库                      | 用途                          | 条件                    |
| ----------------------- | ----------------------------- | ----------------------- |
| `diffusers`             | Stable Diffusion / ControlNet | 启用 Diffusion 风格化时 |
| `accelerate`            | Diffusers 推理调度            | 启用 Diffusion 风格化时 |
| `opencv-contrib-python` | Guided Filter (ximgproc)      | 启用细节注入时          |
| `timm`                  | MiDaS 深度估计                | 启用深度增强时          |

---

## 安装指南

```bash
# 创建环境
conda create -n catoon python=3.10 -y
conda activate catoon

# 安装 PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt
```

---

## 模型权重

### SegFormer（HuggingFace 自动下载）

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    local_files_only=True  # 离线模式
)
```

### Diffusion 模型（可选）

| 模型                 | 用途         | 下载地址                                                             |
| -------------------- | ------------ | -------------------------------------------------------------------- |
| Stable Diffusion 1.5 | 基础生成模型 | [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| ControlNet Canny     | 边缘控制     | [HuggingFace](https://huggingface.co/lllyasviel/sd-controlnet-canny) |

放置位置：`weights/diffusion/`

---

## 常见问题

### cv2.ximgproc 不可用

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

代码已内置 fallback，不影响核心功能。

### CUDA 版本不匹配

```bash
nvidia-smi  # 检查 CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 版本兼容性

| Python | PyTorch | CUDA | transformers | gradio | 状态        |
| ------ | ------- | ---- | ------------ | ------ | ----------- |
| 3.10   | 2.5.x   | 12.1 | 4.57.x       | 6.1.x  | ✅ 当前使用 |
| 3.10   | 2.1.x   | 11.8 | 4.35.x       | 4.x    | ✅ 支持     |

---

## 硬件配置

| 配置等级 | GPU                 | 显存  | 说明                            |
| -------- | ------------------- | ----- | ------------------------------- |
| 最低     | CPU only            | -     | 仅支持传统风格化，速度较慢      |
| 推荐     | GTX 1660 / RTX 3060 | 6GB   | Traditional + SegFormer 流畅    |
| 最佳     | RTX 3080+           | 12GB+ | 支持 Diffusion 风格化、高分辨率 |

> Diffusion 风格化推荐显存 ≥8GB
