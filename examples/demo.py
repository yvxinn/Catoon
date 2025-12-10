#!/usr/bin/env python
"""
Catoon Demo - 命令行演示脚本

使用方法:
    python examples/demo.py [input_image] [output_image]
    
示例:
    python examples/demo.py examples/input.jpg examples/output.jpg
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from PIL import Image

from src.pipeline import load_pipeline


def create_sample_image(width: int = 512, height: int = 384) -> np.ndarray:
    """
    创建一个示例图像（包含天空、建筑、植被、道路）
    
    Returns:
        uint8 RGB 图像
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 天空（上部 30%）- 蓝色渐变
    sky_height = int(height * 0.3)
    for y in range(sky_height):
        ratio = y / sky_height
        img[y, :, 0] = int(135 + 50 * ratio)  # R
        img[y, :, 1] = int(206 - 30 * ratio)  # G
        img[y, :, 2] = int(235 - 20 * ratio)  # B
    
    # 建筑（左侧中部）- 灰色
    building_top = sky_height
    building_bottom = int(height * 0.75)
    building_left = 0
    building_right = int(width * 0.4)
    img[building_top:building_bottom, building_left:building_right, 0] = 150
    img[building_top:building_bottom, building_left:building_right, 1] = 140
    img[building_top:building_bottom, building_left:building_right, 2] = 130
    
    # 窗户
    for wy in range(building_top + 20, building_bottom - 20, 40):
        for wx in range(building_left + 30, building_right - 30, 50):
            img[wy:wy+25, wx:wx+20, 0] = 100
            img[wy:wy+25, wx:wx+20, 1] = 150
            img[wy:wy+25, wx:wx+20, 2] = 200
    
    # 植被（右侧中部）- 绿色
    veg_top = int(height * 0.35)
    veg_bottom = int(height * 0.75)
    veg_left = int(width * 0.45)
    veg_right = width
    for y in range(veg_top, veg_bottom):
        for x in range(veg_left, veg_right):
            noise = np.random.randint(-20, 20)
            img[y, x, 0] = np.clip(34 + noise, 0, 255)
            img[y, x, 1] = np.clip(139 + noise, 0, 255)
            img[y, x, 2] = np.clip(34 + noise, 0, 255)
    
    # 道路（底部）- 深灰色
    road_top = int(height * 0.75)
    img[road_top:, :, 0] = 80
    img[road_top:, :, 1] = 80
    img[road_top:, :, 2] = 80
    
    # 道路标线
    line_y = int(height * 0.85)
    for x in range(0, width, 60):
        img[line_y-2:line_y+2, x:x+30, :] = 255
    
    # 添加随机噪声使图像更自然
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Catoon Demo")
    parser.add_argument("input", nargs="?", help="输入图像路径")
    parser.add_argument("output", nargs="?", default="output.jpg", help="输出图像路径")
    parser.add_argument("--create-sample", action="store_true", help="创建示例图像")
    parser.add_argument("--edge-strength", type=float, default=0.5, help="线稿强度")
    parser.add_argument("--harmonization-strength", type=float, default=0.8, help="协调强度")
    
    args = parser.parse_args()
    
    # 创建示例图像
    if args.create_sample or args.input is None:
        print("创建示例图像...")
        input_image = create_sample_image()
        
        # 保存示例输入
        sample_path = project_root / "examples" / "sample_input.png"
        Image.fromarray(input_image).save(sample_path)
        print(f"示例图像已保存到: {sample_path}")
    else:
        # 加载输入图像
        print(f"加载图像: {args.input}")
        input_image = np.array(Image.open(args.input).convert("RGB"))
    
    print(f"图像尺寸: {input_image.shape}")
    
    # 加载 Pipeline
    print("加载 Pipeline（首次运行需要下载模型）...")
    pipe = load_pipeline()
    
    # 设置参数
    ui_params = {
        "edge_strength": args.edge_strength,
        "harmonization_enabled": True,
        "harmonization_strength": args.harmonization_strength,
        "gamma": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
    }
    
    # 处理
    print("处理图像...")
    output_image = pipe.process(input_image, ui_params)
    
    # 保存输出
    output_path = project_root / "examples" / args.output if not Path(args.output).is_absolute() else Path(args.output)
    Image.fromarray(output_image).save(output_path)
    print(f"输出已保存到: {output_path}")
    
    print("完成!")


if __name__ == "__main__":
    main()

