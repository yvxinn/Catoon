"""
Gradio UI - 交互式卡通化界面

提供可视化的参数调整和实时预览。
支持实时调整（不重新推理）的参数。
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np

# Pipeline 延迟导入
pipeline = None

# 缓存中间结果（避免重复推理）
_cache = {
    "image_hash": None,
    "ctx": None,
    "seg_out": None,
    "face_mask": None,
    "candidates": None,
    "trad_params": None,  # (k, smooth_method)
}


def get_pipeline():
    """懒加载 Pipeline"""
    global pipeline
    if pipeline is None:
        from src.pipeline import load_pipeline
        pipeline = load_pipeline()
    return pipeline


def _compute_image_hash(image: np.ndarray) -> str:
    """计算图像哈希（用于缓存判断）"""
    return str(hash(image.tobytes()))


def _needs_full_recompute(
    image: np.ndarray,
    traditional_k: int,
    traditional_smooth_method: str
) -> bool:
    """判断是否需要完整重新计算"""
    if image is None:
        return False
    
    img_hash = _compute_image_hash(image)
    trad_params = (traditional_k, traditional_smooth_method)
    
    # 图像或传统风格化参数变化时需要重新计算
    if _cache["image_hash"] != img_hash or _cache["trad_params"] != trad_params:
        return True
    return False


def full_compute(
    image: np.ndarray,
    traditional_smooth_method: str,
    traditional_k: int
):
    """
    完整计算（需要模型推理）
    缓存：预处理结果、分割结果、人脸检测、风格候选
    """
    if image is None:
        return
    
    pipe = get_pipeline()
    img_hash = _compute_image_hash(image)
    trad_params = (traditional_k, traditional_smooth_method)
    
    # 检查是否需要重新计算
    if _cache["image_hash"] == img_hash and _cache["trad_params"] == trad_params:
        return  # 使用缓存
    
    print("[Pipeline] 执行完整计算...")
    
    # A. 预处理
    ctx = pipe.preprocessor.process(image)
    
    # B. 语义分析
    seg_out = pipe.segmenter.predict(ctx.image_f32)
    face_mask = None
    if pipe.face_detector:
        face_mask = pipe.face_detector.detect(ctx.image_u8)
    
    # C. 风格候选生成
    ui_params = {
        "traditional_k": traditional_k,
        "traditional_smooth_method": traditional_smooth_method
    }
    candidates = pipe._get_or_build_candidates(ctx, ui_params)
    
    # 更新缓存
    _cache["image_hash"] = img_hash
    _cache["ctx"] = ctx
    _cache["seg_out"] = seg_out
    _cache["face_mask"] = face_mask
    _cache["candidates"] = candidates
    _cache["trad_params"] = trad_params
    
    print("[Pipeline] 完整计算完成，已缓存中间结果")


def realtime_render(
    # 融合
    fusion_method: str,
    fusion_blur_kernel: int,
    # 协调
    harmonization_enabled: bool,
    harmonization_reference: str,
    harmonization_strength: float,
    # 线稿
    edge_strength: float,
    line_engine: str,
    line_width: int,
    canny_low: int,
    canny_high: int,
    xdog_sigma: float,
    xdog_k: float,
    xdog_p: float,
    # 细节增强
    detail_enhance_enabled: bool,
    detail_strength: float,
    # 色调
    gamma: float,
    contrast: float,
    saturation: float,
    brightness: float,
    # 人脸
    face_protect_enabled: bool,
    face_protect_mode: str,
    face_gan_weight_max: float,
    # 区域风格
    sky_style: str,
    person_style: str,
    building_style: str,
    vegetation_style: str,
    road_style: str,
    water_style: str,
    others_style: str,
) -> np.ndarray | None:
    """
    实时渲染（不重新推理，直接使用缓存）
    """
    if _cache["candidates"] is None:
        return None
    
    pipe = get_pipeline()
    ctx = _cache["ctx"]
    seg_out = _cache["seg_out"]
    face_mask = _cache["face_mask"]
    candidates = _cache["candidates"]
    
    # 构建 UI 参数
    ui_params = {
        "fusion_method": fusion_method,
        "fusion_blur_kernel": fusion_blur_kernel,
        "harmonization_enabled": harmonization_enabled,
        "harmonization_reference": harmonization_reference,
        "harmonization_strength": harmonization_strength,
        "edge_strength": edge_strength,
        "line_engine": line_engine,
        "line_width": line_width,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "xdog_sigma": xdog_sigma,
        "xdog_k": xdog_k,
        "xdog_p": xdog_p,
        "detail_enhance_enabled": detail_enhance_enabled,
        "detail_strength": detail_strength,
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
            "ROAD": {"style": road_style},
            "WATER": {"style": water_style},
            "OTHERS": {"style": others_style},
        }
    }
    
    # D. 语义路由（轻量）
    routing = pipe.router.route(
        semantic_masks=seg_out.semantic_masks,
        face_mask=face_mask,
        ui_overrides=ui_params
    )
    
    # E. 区域融合（轻量）
    fused = pipe.fuser.fuse(
        candidates=candidates,
        routing=routing,
        seg_out=seg_out,
        method=fusion_method,
        blur_kernel=fusion_blur_kernel
    )
    
    # F. 全局协调（轻量）
    if harmonization_enabled:
        ref = pipe.harmonizer.pick_reference(
            candidates, seg_out, ui_params, pipe.cfg.harmonization
        )
        fused = pipe.harmonizer.match_and_adjust(fused, ref, ui_params)
    
    # G. 线稿叠加（轻量）
    if edge_strength > 1e-3:
        edges = pipe.lineart.extract(ctx.image_u8, ui_params)
        fused = pipe.lineart.overlay(fused, edges, edge_strength, ui_params)
    
    # G2. 细节增强（轻量）
    if detail_enhance_enabled:
        fused = pipe.lineart.enhance_detail(fused, ctx.image_f32, detail_strength)
    
    # 色调调整（轻量）
    fused = apply_tone_adjustment(fused, gamma, contrast, saturation, brightness)
    
    # 后处理
    out_u8 = pipe.preprocessor.postprocess(fused, ctx)
    return out_u8


def apply_tone_adjustment(
    image: np.ndarray,
    gamma: float,
    contrast: float,
    saturation: float,
    brightness: float
) -> np.ndarray:
    """应用色调调整"""
    import cv2
    
    result = image.copy()
    
    # Gamma
    if abs(gamma - 1.0) > 0.01:
        result = np.power(result, 1.0 / gamma)
    
    # Contrast
    if abs(contrast - 1.0) > 0.01:
        result = (result - 0.5) * contrast + 0.5
    
    # Brightness
    if abs(brightness) > 0.1:
        result = result + brightness / 255.0
    
    # Saturation
    if abs(saturation - 1.0) > 0.01:
        # 转换到 HSV
        result_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(result_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = result_u8.astype(np.float32) / 255.0
    
    return np.clip(result, 0, 1).astype(np.float32)


def process_image(
    image: np.ndarray,
    # 需要重新推理的参数
    traditional_smooth_method: str,
    traditional_k: int,
    # 实时参数
    fusion_method: str,
    fusion_blur_kernel: int,
    harmonization_enabled: bool,
    harmonization_reference: str,
    harmonization_strength: float,
    edge_strength: float,
    line_engine: str,
    line_width: int,
    canny_low: int,
    canny_high: int,
    xdog_sigma: float,
    xdog_k: float,
    xdog_p: float,
    detail_enhance_enabled: bool,
    detail_strength: float,
    gamma: float,
    contrast: float,
    saturation: float,
    brightness: float,
    face_protect_enabled: bool,
    face_protect_mode: str,
    face_gan_weight_max: float,
    sky_style: str,
    person_style: str,
    building_style: str,
    vegetation_style: str,
    road_style: str,
    water_style: str,
    others_style: str,
) -> np.ndarray | None:
    """完整处理（上传新图像或更改重计算参数时调用）"""
    if image is None:
        return None
    
    # 执行完整计算（会自动判断是否需要）
    full_compute(image, traditional_smooth_method, traditional_k)
    
    # 实时渲染
    return realtime_render(
        fusion_method, fusion_blur_kernel,
        harmonization_enabled, harmonization_reference, harmonization_strength,
        edge_strength, line_engine, line_width,
        canny_low, canny_high, xdog_sigma, xdog_k, xdog_p,
        detail_enhance_enabled, detail_strength,
        gamma, contrast, saturation, brightness,
        face_protect_enabled, face_protect_mode, face_gan_weight_max,
        sky_style, person_style, building_style, vegetation_style,
        road_style, water_style, others_style,
    )


def create_ui():
    """创建 Gradio UI"""
    
    style_choices = ["Traditional", "Hayao", "Shinkai", "Paprika"]
    semantic_buckets = ["SKY", "PERSON", "BUILDING", "VEGETATION", "ROAD", "WATER", "OTHERS"]
    
    with gr.Blocks(title="Catoon - 语义感知可控卡通化") as demo:
        
        gr.Markdown("""
        # 🎨 Catoon - 语义感知可控卡通化框架
        
        上传图像后，调整参数可**实时预览**效果！
        
        > 💡 **实时调整**：融合、线稿、色调、区域风格等参数更改后立即生效  
        > 🔄 **重新计算**：仅上传新图像或更改风格化参数时重新推理
        """)
        
        with gr.Row():
            # 左侧：输入和输出
            with gr.Column(scale=2):
                input_image = gr.Image(label="📷 输入图像", type="numpy")
                output_image = gr.Image(label="🖼️ 输出结果", type="numpy")
                
                with gr.Row():
                    process_btn = gr.Button("🚀 处理图像", variant="primary", size="lg")
                    realtime_toggle = gr.Checkbox(
                        value=True, 
                        label="⚡ 实时预览",
                        info="开启后调整参数立即更新"
                    )
            
            # 右侧：参数控制
            with gr.Column(scale=1):
                
                # ========== 风格化设置（需要重新计算）==========
                with gr.Accordion("🖌️ 风格化设置 (更改后需重新计算)", open=False):
                    gr.Markdown("⚠️ *更改这些参数需要点击「处理图像」按钮*")
                    traditional_smooth_method = gr.Radio(
                        choices=["bilateral", "edge_preserving", "mean_shift"],
                        value="bilateral",
                        label="平滑方法"
                    )
                    traditional_k = gr.Slider(
                        4, 48, value=16, step=4,
                        label="颜色量化 K"
                    )
                
                # ========== 以下参数支持实时调整 ==========
                gr.Markdown("---\n**以下参数支持实时调整** ⚡")
                
                # ========== 融合设置 ==========
                with gr.Accordion("🔀 融合设置", open=True):
                    fusion_method = gr.Radio(
                        choices=["soft_mask", "laplacian_pyramid", "poisson"],
                        value="soft_mask",
                        label="融合方法"
                    )
                    fusion_blur_kernel = gr.Slider(
                        5, 51, value=21, step=2,
                        label="模糊核大小"
                    )
                
                # ========== 区域风格 ==========
                with gr.Accordion("🗺️ 区域风格", open=True):
                    sky_style = gr.Dropdown(choices=style_choices, value="Shinkai", label="☁️ 天空")
                    person_style = gr.Dropdown(choices=style_choices, value="Traditional", label="👤 人物")
                    building_style = gr.Dropdown(choices=style_choices, value="Traditional", label="🏠 建筑")
                    vegetation_style = gr.Dropdown(choices=style_choices, value="Hayao", label="🌳 植被")
                    road_style = gr.Dropdown(choices=style_choices, value="Traditional", label="🛤️ 道路")
                    water_style = gr.Dropdown(choices=style_choices, value="Shinkai", label="🌊 水体")
                    others_style = gr.Dropdown(choices=style_choices, value="Traditional", label="📦 其他")
                
                # ========== 线稿设置 ==========
                with gr.Accordion("✏️ 线稿设置", open=True):
                    edge_strength = gr.Slider(0, 1, value=0.5, label="线稿强度")
                    line_engine = gr.Radio(choices=["canny", "xdog"], value="canny", label="线稿引擎")
                    line_width = gr.Slider(1, 5, value=1, step=1, label="线条宽度")
                    
                    with gr.Group():
                        gr.Markdown("**Canny 参数**")
                        canny_low = gr.Slider(50, 150, value=100, label="低阈值")
                        canny_high = gr.Slider(100, 300, value=200, label="高阈值")
                    
                    with gr.Group():
                        gr.Markdown("**XDoG 参数**")
                        xdog_sigma = gr.Slider(0.1, 2.0, value=0.5, label="Sigma")
                        xdog_k = gr.Slider(1.0, 3.0, value=1.6, label="K")
                        xdog_p = gr.Slider(5.0, 50.0, value=19.0, label="P")
                
                # ========== 全局协调 ==========
                with gr.Accordion("🎨 全局协调", open=False):
                    harmonization_enabled = gr.Checkbox(value=True, label="启用直方图匹配")
                    harmonization_reference = gr.Dropdown(
                        choices=semantic_buckets + ["auto"],
                        value="SKY",
                        label="参考区域"
                    )
                    harmonization_strength = gr.Slider(0, 1, value=0.8, label="匹配强度")
                
                # ========== 细节增强 ==========
                with gr.Accordion("🔍 细节增强", open=False):
                    detail_enhance_enabled = gr.Checkbox(value=False, label="启用 Guided Filter")
                    detail_strength = gr.Slider(0, 1, value=0.5, label="增强强度")
                
                # ========== 色调调整 ==========
                with gr.Accordion("🌈 色调调整", open=False):
                    gamma = gr.Slider(0.5, 2.0, value=1.0, label="Gamma")
                    contrast = gr.Slider(0.5, 1.5, value=1.0, label="对比度")
                    saturation = gr.Slider(0.5, 1.5, value=1.0, label="饱和度")
                    brightness = gr.Slider(-50, 50, value=0, label="亮度")
                
                # ========== 人脸保护 ==========
                with gr.Accordion("👤 人脸保护", open=False):
                    face_protect_enabled = gr.Checkbox(value=True, label="启用人脸保护")
                    face_protect_mode = gr.Radio(
                        choices=["protect", "blend", "full_style"],
                        value="protect",
                        label="保护模式"
                    )
                    face_gan_weight_max = gr.Slider(0, 1, value=0.3, label="GAN 权重上限")
        
        # 所有输入参数列表
        all_inputs = [
            input_image,
            traditional_smooth_method, traditional_k,
            fusion_method, fusion_blur_kernel,
            harmonization_enabled, harmonization_reference, harmonization_strength,
            edge_strength, line_engine, line_width,
            canny_low, canny_high, xdog_sigma, xdog_k, xdog_p,
            detail_enhance_enabled, detail_strength,
            gamma, contrast, saturation, brightness,
            face_protect_enabled, face_protect_mode, face_gan_weight_max,
            sky_style, person_style, building_style, vegetation_style,
            road_style, water_style, others_style,
        ]
        
        # 实时调整参数（不包含 input_image 和 traditional_* ）
        realtime_inputs = all_inputs[3:]  # 跳过 image 和 traditional 参数
        
        # 点击按钮处理
        process_btn.click(
            fn=process_image,
            inputs=all_inputs,
            outputs=output_image
        )
        
        # 图像上传时自动处理
        input_image.change(
            fn=process_image,
            inputs=all_inputs,
            outputs=output_image
        )
        
        # 实时预览函数
        def realtime_update(*args):
            """实时更新（仅当缓存存在时）"""
            if _cache["candidates"] is None:
                return None
            return realtime_render(*args)
        
        # 为实时参数绑定 change 事件
        realtime_components = [
            fusion_method, fusion_blur_kernel,
            harmonization_enabled, harmonization_reference, harmonization_strength,
            edge_strength, line_engine, line_width,
            canny_low, canny_high, xdog_sigma, xdog_k, xdog_p,
            detail_enhance_enabled, detail_strength,
            gamma, contrast, saturation, brightness,
            face_protect_enabled, face_protect_mode, face_gan_weight_max,
            sky_style, person_style, building_style, vegetation_style,
            road_style, water_style, others_style,
        ]
        
        for component in realtime_components:
            component.change(
                fn=realtime_update,
                inputs=realtime_components,
                outputs=output_image
            )
        
        gr.Markdown("""
        ---
        ### ⚡ 实时预览说明
        
        | 参数类型 | 行为 |
        |---------|------|
        | **风格化参数** | 需点击「处理图像」重新计算 |
        | **其他参数** | 调整后立即更新预览 |
        
        **风格说明**：`Traditional` 双边滤波 | `Hayao` 宫崎骏 | `Shinkai` 新海诚 | `Paprika` 今敏
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
