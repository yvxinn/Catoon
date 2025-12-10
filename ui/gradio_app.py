"""
Gradio UI - 交互式卡通化界面

提供可视化的参数调整和实时预览。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np

# Pipeline 延迟导入（避免启动时加载模型）
pipeline = None


def get_pipeline():
    """懒加载 Pipeline"""
    global pipeline
    if pipeline is None:
        from src.pipeline import load_pipeline
        pipeline = load_pipeline()
    return pipeline


def process_image(
    image: np.ndarray,
    # 融合方法
    fusion_method: str,
    # 全局协调
    harmonization_enabled: bool,
    harmonization_reference: str,
    harmonization_strength: float,
    # 线稿
    edge_strength: float,
    line_engine: str,
    # 全局色调
    gamma: float,
    contrast: float,
    saturation: float,
    brightness: float,
    # 人脸保护
    face_protect_enabled: bool,
    face_protect_mode: str,
    face_gan_weight_max: float,
    # 区域风格（简化版，后续可扩展）
    sky_style: str,
    person_style: str,
    building_style: str,
    vegetation_style: str,
) -> np.ndarray:
    """处理图像的主函数"""
    if image is None:
        return None
    
    pipe = get_pipeline()
    
    # 构建 UI 参数
    ui_params = {
        "fusion_method": fusion_method,
        "harmonization_enabled": harmonization_enabled,
        "harmonization_reference": harmonization_reference,
        "harmonization_strength": harmonization_strength,
        "edge_strength": edge_strength,
        "line_engine": line_engine,
        "gamma": gamma,
        "contrast": contrast,
        "saturation": saturation,
        "brightness": brightness,
        "face_protect_enabled": face_protect_enabled,
        "face_protect_mode": face_protect_mode,
        "face_gan_weight_max": face_gan_weight_max,
        "region_overrides": {
            "SKY": {"style": sky_style},
            "PERSON": {"style": person_style},
            "BUILDING": {"style": building_style},
            "VEGETATION": {"style": vegetation_style},
        }
    }
    
    # 处理图像
    result = pipe.process(image, ui_params)
    return result


def create_ui():
    """创建 Gradio UI"""
    
    # 可用风格列表
    style_choices = ["Hayao", "Shinkai", "Paprika", "Traditional"]
    
    with gr.Blocks(
        title="Catoon - 语义感知可控卡通化",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🎨 Catoon - 语义感知可控卡通化框架
        
        上传图像，为不同语义区域选择不同的卡通风格！
        """)
        
        with gr.Row():
            # 左侧：输入和输出
            with gr.Column(scale=2):
                input_image = gr.Image(label="输入图像", type="numpy")
                output_image = gr.Image(label="输出结果", type="numpy")
                process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
            
            # 右侧：参数控制
            with gr.Column(scale=1):
                with gr.Accordion("🔀 融合设置", open=True):
                    fusion_method = gr.Radio(
                        choices=["soft_mask", "laplacian_pyramid", "poisson"],
                        value="soft_mask",
                        label="融合方法"
                    )
                
                with gr.Accordion("🎨 全局协调", open=True):
                    harmonization_enabled = gr.Checkbox(value=True, label="启用直方图匹配")
                    harmonization_reference = gr.Dropdown(
                        choices=["SKY", "PERSON", "BUILDING", "auto"],
                        value="SKY",
                        label="参考区域"
                    )
                    harmonization_strength = gr.Slider(0, 1, value=0.8, label="匹配强度")
                
                with gr.Accordion("✏️ 线稿设置", open=True):
                    edge_strength = gr.Slider(0, 1, value=0.5, label="线稿强度")
                    line_engine = gr.Radio(
                        choices=["canny", "xdog"],
                        value="canny",
                        label="线稿引擎"
                    )
                
                with gr.Accordion("🌈 色调调整", open=False):
                    gamma = gr.Slider(0.5, 2.0, value=1.0, label="Gamma")
                    contrast = gr.Slider(0.5, 1.5, value=1.0, label="对比度")
                    saturation = gr.Slider(0.5, 1.5, value=1.0, label="饱和度")
                    brightness = gr.Slider(-50, 50, value=0, label="亮度")
                
                with gr.Accordion("👤 人脸保护", open=False):
                    face_protect_enabled = gr.Checkbox(value=True, label="启用人脸保护")
                    face_protect_mode = gr.Radio(
                        choices=["protect", "blend", "full_style"],
                        value="protect",
                        label="保护模式"
                    )
                    face_gan_weight_max = gr.Slider(0, 1, value=0.3, label="GAN权重上限")
                
                with gr.Accordion("🗺️ 区域风格", open=True):
                    sky_style = gr.Dropdown(choices=style_choices, value="Shinkai", label="天空")
                    person_style = gr.Dropdown(choices=style_choices, value="Hayao", label="人物")
                    building_style = gr.Dropdown(choices=style_choices, value="Traditional", label="建筑")
                    vegetation_style = gr.Dropdown(choices=style_choices, value="Paprika", label="植被")
        
        # 绑定处理函数
        process_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                fusion_method,
                harmonization_enabled, harmonization_reference, harmonization_strength,
                edge_strength, line_engine,
                gamma, contrast, saturation, brightness,
                face_protect_enabled, face_protect_mode, face_gan_weight_max,
                sky_style, person_style, building_style, vegetation_style,
            ],
            outputs=output_image
        )
        
        gr.Markdown("""
        ---
        **提示**：
        - 首次处理可能需要加载模型，请耐心等待
        - 建议先使用默认参数，再根据效果微调
        - 人脸保护可防止人物面部过度风格化
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

