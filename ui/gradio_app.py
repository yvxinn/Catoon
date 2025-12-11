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
    "original_image": None,  # 原图用于遮罩可视化
    "active_masks": set(),   # 当前激活的语义遮罩（用于叠加）
}

# 语义区域颜色映射（更鲜艳的颜色）
SEMANTIC_COLORS = {
    "SKY": (0, 150, 255),         # 亮蓝色
    "PERSON": (255, 50, 150),     # 亮粉色
    "BUILDING": (255, 150, 0),    # 橙色
    "VEGETATION": (0, 255, 100),  # 亮绿色
    "ROAD": (128, 128, 128),      # 灰色
    "WATER": (0, 200, 255),       # 青色
    "OTHERS": (255, 255, 0),      # 黄色
    "FACE": (255, 0, 100),        # 玫红色
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
    _cache["original_image"] = image.copy()
    _cache["active_masks"] = set()  # 重置激活的遮罩
    
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
    # 区域风格（风格 + 强度 + K）
    sky_style: str, sky_strength: float, sky_k: int,
    person_style: str, person_strength: float, person_k: int,
    building_style: str, building_strength: float, building_k: int,
    vegetation_style: str, vegetation_strength: float, vegetation_k: int,
    road_style: str, road_strength: float, road_k: int,
    water_style: str, water_strength: float, water_k: int,
    others_style: str, others_strength: float, others_k: int,
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
    
    # 构建 UI 参数（包含区域级 strength 和 k）
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
            "SKY": {"style": sky_style, "strength": sky_strength, "k": int(sky_k)},
            "PERSON": {"style": person_style, "strength": person_strength, "k": int(person_k)},
            "BUILDING": {"style": building_style, "strength": building_strength, "k": int(building_k)},
            "VEGETATION": {"style": vegetation_style, "strength": vegetation_strength, "k": int(vegetation_k)},
            "ROAD": {"style": road_style, "strength": road_strength, "k": int(road_k)},
            "WATER": {"style": water_style, "strength": water_strength, "k": int(water_k)},
            "OTHERS": {"style": others_style, "strength": others_strength, "k": int(others_k)},
        }
    }
    
    # D. 语义路由（轻量）
    routing = pipe.router.route(
        semantic_masks=seg_out.semantic_masks,
        face_mask=face_mask,
        ui_overrides=ui_params
    )
    
    # C2. 区域级风格化（按需生成，带缓存）
    region_candidates = pipe.region_stylizer.generate_region_styles(
        image_f32=ctx.image_f32,
        image_hash=ctx.image_hash,
        seg_out=seg_out,
        region_configs=routing.region_configs,
        global_candidates=candidates
    )
    
    # E. 区域融合（轻量）- 传递原图和区域候选
    fused = pipe.fuser.fuse(
        candidates=candidates,
        routing=routing,
        seg_out=seg_out,
        method=fusion_method,
        blur_kernel=fusion_blur_kernel,
        original_image=ctx.image_f32,
        region_candidates=region_candidates
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


def visualize_semantic_mask(bucket: str, toggle: bool = True) -> tuple[np.ndarray | None, str]:
    """
    可视化指定语义区域的遮罩（支持叠加多个区域）
    
    Args:
        bucket: 语义桶名称 (SKY, PERSON, etc.) 或 "FACE" 或 "NONE"
        toggle: 是否切换该区域的显示状态
    
    Returns:
        (叠加遮罩后的图像, 覆盖率信息)
    """
    if _cache["original_image"] is None or _cache["seg_out"] is None:
        return None, "请先上传并处理图像"
    
    # 处理 NONE（清除所有遮罩）
    if bucket == "NONE":
        _cache["active_masks"] = set()
        return _cache["original_image"].copy(), "显示原图"
    
    # 切换该区域的激活状态
    if toggle:
        if bucket in _cache["active_masks"]:
            _cache["active_masks"].discard(bucket)
        else:
            _cache["active_masks"].add(bucket)
    
    # 如果没有激活的遮罩，返回原图
    if not _cache["active_masks"]:
        return _cache["original_image"].copy(), "点击区域按钮查看遮罩"
    
    import cv2
    
    # 获取原图
    original = _cache["original_image"].copy()
    H, W = original.shape[:2]
    result = original.astype(np.float32)
    
    info_parts = []
    
    # 叠加所有激活的遮罩
    for active_bucket in _cache["active_masks"]:
        # 获取遮罩
        if active_bucket == "FACE":
            if _cache["face_mask"] is None:
                continue
            mask = _cache["face_mask"]
        else:
            seg_out = _cache["seg_out"]
            if active_bucket not in seg_out.semantic_masks:
                continue
            mask = seg_out.semantic_masks[active_bucket]
        
        # 调整遮罩尺寸到原图大小
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 计算覆盖率
        coverage = mask.mean() * 100
        info_parts.append(f"{active_bucket}: {coverage:.1f}%")
        
        # 创建彩色遮罩
        color = SEMANTIC_COLORS.get(active_bucket, (255, 255, 0))
        colored_mask = np.zeros((H, W, 3), dtype=np.float32)
        colored_mask[:, :] = color
        
        # 叠加遮罩（半透明）
        alpha = 0.5
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = result * (1 - mask_3d * alpha) + colored_mask * mask_3d * alpha
        
        # 添加边界轮廓
        mask_u8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_u8 = np.clip(result, 0, 255).astype(np.uint8)
        cv2.drawContours(result_u8, contours, -1, color, 2)
        result = result_u8.astype(np.float32)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    info = "🎯 " + " | ".join(info_parts) if info_parts else "无激活区域"
    
    return result, info


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
    # 区域风格（风格 + 强度 + K）
    sky_style: str, sky_strength: float, sky_k: int,
    person_style: str, person_strength: float, person_k: int,
    building_style: str, building_strength: float, building_k: int,
    vegetation_style: str, vegetation_strength: float, vegetation_k: int,
    road_style: str, road_strength: float, road_k: int,
    water_style: str, water_strength: float, water_k: int,
    others_style: str, others_strength: float, others_k: int,
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
        sky_style, sky_strength, sky_k,
        person_style, person_strength, person_k,
        building_style, building_strength, building_k,
        vegetation_style, vegetation_strength, vegetation_k,
        road_style, road_strength, road_k,
        water_style, water_strength, water_k,
        others_style, others_strength, others_k,
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
                # 语义遮罩可视化按钮（移到顶部）
                gr.Markdown("**🔍 点击切换语义区域遮罩** *(可叠加多个)*")
                with gr.Row():
                    btn_none = gr.Button("🔄 清除", size="sm")
                    btn_sky = gr.Button("☁️ 天空", size="sm", variant="secondary")
                    btn_person = gr.Button("👤 人物", size="sm", variant="secondary")
                    btn_face = gr.Button("😊 人脸", size="sm", variant="secondary")
                    btn_building = gr.Button("🏠 建筑", size="sm", variant="secondary")
                with gr.Row():
                    btn_vegetation = gr.Button("🌳 植被", size="sm", variant="secondary")
                    btn_road = gr.Button("🛤️ 道路", size="sm", variant="secondary")
                    btn_water = gr.Button("🌊 水体", size="sm", variant="secondary")
                    btn_others = gr.Button("📦 其他", size="sm", variant="secondary")
                
                mask_info = gr.Textbox(label="", value="上传图像后点击按钮查看语义区域", show_label=False, max_lines=1)
                
                # 使用单独的预览组件，不影响 input_image
                with gr.Row():
                    input_image = gr.Image(label="📷 输入图像", type="numpy")
                    mask_preview = gr.Image(label="🔍 语义遮罩预览", type="numpy")
                
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
                        label="平滑方法",
                        info="bilateral: 双边滤波，保边效果好 | edge_preserving: OpenCV边缘保持 | mean_shift: 均值漂移，色块更明显"
                    )
                    traditional_k = gr.Slider(
                        4, 48, value=16, step=4,
                        label="颜色量化 K",
                        info="K值越大颜色越丰富，越小色块越明显（推荐8-24）"
                    )
                
                # ========== 以下参数支持实时调整 ==========
                gr.Markdown("---\n**以下参数支持实时调整** ⚡")
                
                # ========== 融合设置 ==========
                with gr.Accordion("🔀 融合设置", open=True):
                    fusion_method = gr.Radio(
                        choices=["soft_mask", "laplacian_pyramid", "poisson"],
                        value="soft_mask",
                        label="融合方法",
                        info="soft_mask: 快速模糊融合 | laplacian_pyramid: 多尺度融合，接缝更自然 | poisson: 泊松融合（实验性）"
                    )
                    fusion_blur_kernel = gr.Slider(
                        5, 51, value=21, step=2,
                        label="模糊核大小",
                        info="控制区域边界的过渡宽度，值越大过渡越平滑"
                    )
                
                # ========== 区域风格 ==========
                with gr.Accordion("🗺️ 区域风格", open=True):
                    gr.Markdown("*每个区域可独立设置：风格、强度、K值*")
                    
                    # 天空
                    with gr.Group():
                        with gr.Row():
                            sky_style = gr.Dropdown(choices=style_choices, value="Shinkai", label="☁️ 天空",
                                info="推荐 Shinkai", scale=2)
                            sky_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1,
                                info="0=原图，1=完全风格化")
                            sky_k = gr.Slider(4, 64, value=16, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                    
                    # 人物
                    with gr.Group():
                        with gr.Row():
                            person_style = gr.Dropdown(choices=style_choices, value="Traditional", label="👤 人物",
                                info="推荐 Traditional", scale=2)
                            person_strength = gr.Slider(0, 1, value=0.7, label="强度", scale=1,
                                info="人物建议0.5-0.8")
                            person_k = gr.Slider(4, 64, value=20, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                    
                    # 建筑
                    with gr.Group():
                        with gr.Row():
                            building_style = gr.Dropdown(choices=style_choices, value="Traditional", label="🏠 建筑",
                                info="建筑物风格", scale=2)
                            building_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1)
                            building_k = gr.Slider(4, 64, value=16, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                    
                    # 植被
                    with gr.Group():
                        with gr.Row():
                            vegetation_style = gr.Dropdown(choices=style_choices, value="Hayao", label="🌳 植被",
                                info="推荐 Hayao", scale=2)
                            vegetation_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1)
                            vegetation_k = gr.Slider(4, 64, value=24, step=2, label="K", scale=1,
                                info="Traditional 专用，植被建议 K 大一些，范围 4-64")
                    
                    # 道路
                    with gr.Group():
                        with gr.Row():
                            road_style = gr.Dropdown(choices=style_choices, value="Traditional", label="🛤️ 道路",
                                info="道路/地面风格", scale=2)
                            road_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1)
                            road_k = gr.Slider(4, 64, value=12, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                    
                    # 水体
                    with gr.Group():
                        with gr.Row():
                            water_style = gr.Dropdown(choices=style_choices, value="Shinkai", label="🌊 水体",
                                info="推荐 Shinkai", scale=2)
                            water_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1)
                            water_k = gr.Slider(4, 64, value=16, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                    
                    # 其他
                    with gr.Group():
                        with gr.Row():
                            others_style = gr.Dropdown(choices=style_choices, value="Traditional", label="📦 其他",
                                info="未分类区域", scale=2)
                            others_strength = gr.Slider(0, 1, value=1.0, label="强度", scale=1)
                            others_k = gr.Slider(4, 64, value=16, step=2, label="K", scale=1,
                                info="Traditional 专用，范围 4-64")
                
                # ========== 线稿设置 ==========
                with gr.Accordion("✏️ 线稿设置", open=True):
                    edge_strength = gr.Slider(0, 1, value=0.5, label="线稿强度",
                        info="0=无线稿，1=最强线稿，推荐0.3-0.6")
                    line_engine = gr.Radio(choices=["canny", "xdog"], value="canny", label="线稿引擎",
                        info="canny: 经典边缘检测，稳定 | xdog: 艺术风格线条，更有手绘感")
                    line_width = gr.Slider(0.5, 4, value=1, step=0.25, label="线条宽度",
                        info="线条粗细更精细：0.5=极细，2=中等，4=较粗（内部会取整）")
                    
                    with gr.Group():
                        gr.Markdown("**Canny 参数**")
                        canny_low = gr.Slider(50, 150, value=100, label="低阈值",
                            info="边缘检测低阈值，值越低检测到的边缘越多")
                        canny_high = gr.Slider(100, 300, value=200, label="高阈值",
                            info="边缘检测高阈值，值越高只保留强边缘")
                    
                    with gr.Group():
                        gr.Markdown("**XDoG 参数**")
                        xdog_sigma = gr.Slider(0.1, 2.0, value=0.5, label="Sigma",
                            info="高斯模糊程度，值越大线条越粗犷")
                        xdog_k = gr.Slider(1.0, 3.0, value=1.6, label="K",
                            info="两个高斯核的比例，影响边缘检测范围")
                        xdog_p = gr.Slider(5.0, 50.0, value=19.0, label="P",
                            info="锐化程度，值越大线条对比度越高")
                
                # ========== 全局协调 ==========
                with gr.Accordion("🎨 全局协调", open=False):
                    harmonization_enabled = gr.Checkbox(value=True, label="启用直方图匹配",
                        info="统一各区域的色调，减少拼接感")
                    harmonization_reference = gr.Dropdown(
                        choices=semantic_buckets + ["auto"],
                        value="SKY",
                        label="参考区域",
                        info="以哪个区域的色调为基准进行统一"
                    )
                    harmonization_strength = gr.Slider(0, 1, value=0.8, label="匹配强度",
                        info="色调统一的程度，0=不统一，1=完全统一")
                
                # ========== 细节增强 ==========
                with gr.Accordion("🔍 细节增强", open=False):
                    detail_enhance_enabled = gr.Checkbox(value=False, label="启用 Guided Filter",
                        info="使用导向滤波增强图像细节和纹理")
                    detail_strength = gr.Slider(0, 1, value=0.5, label="增强强度",
                        info="细节增强程度，过高可能产生噪点")
                
                # ========== 色调调整 ==========
                with gr.Accordion("🌈 色调调整", open=False):
                    gamma = gr.Slider(0.5, 2.0, value=1.0, label="Gamma",
                        info="<1 变亮，>1 变暗，调整整体明暗")
                    contrast = gr.Slider(0.5, 1.5, value=1.0, label="对比度",
                        info="<1 降低对比度，>1 增强对比度")
                    saturation = gr.Slider(0.5, 1.5, value=1.0, label="饱和度",
                        info="<1 降低饱和度（偏灰），>1 增强饱和度（更鲜艳）")
                    brightness = gr.Slider(-50, 50, value=0, label="亮度",
                        info="直接增减亮度值，负值变暗，正值变亮")
                
                # ========== 人脸保护 ==========
                with gr.Accordion("👤 人脸保护", open=False):
                    face_protect_enabled = gr.Checkbox(value=True, label="启用人脸保护",
                        info="保护人脸区域不被过度风格化")
                    face_protect_mode = gr.Radio(
                        choices=["protect", "blend", "full_style"],
                        value="protect",
                        label="保护模式",
                        info="protect: 最大保护 | blend: 轻微风格化 | full_style: 无保护"
                    )
                    face_gan_weight_max = gr.Slider(0, 1, value=0.3, label="GAN 权重上限",
                        info="人脸区域允许的最大 GAN 风格化强度"
                    )
        
        # 所有输入参数列表（包含区域级 strength 和 k）
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
            sky_style, sky_strength, sky_k,
            person_style, person_strength, person_k,
            building_style, building_strength, building_k,
            vegetation_style, vegetation_strength, vegetation_k,
            road_style, road_strength, road_k,
            water_style, water_strength, water_k,
            others_style, others_strength, others_k,
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
            sky_style, sky_strength, sky_k,
            person_style, person_strength, person_k,
            building_style, building_strength, building_k,
            vegetation_style, vegetation_strength, vegetation_k,
            road_style, road_strength, road_k,
            water_style, water_strength, water_k,
            others_style, others_strength, others_k,
        ]
        
        for component in realtime_components:
            component.change(
                fn=realtime_update,
                inputs=realtime_components,
                outputs=output_image
            )
        
        # 语义遮罩可视化按钮绑定（更新单独的预览组件，不影响输入图像）
        btn_none.click(lambda: visualize_semantic_mask("NONE"), outputs=[mask_preview, mask_info])
        btn_sky.click(lambda: visualize_semantic_mask("SKY"), outputs=[mask_preview, mask_info])
        btn_person.click(lambda: visualize_semantic_mask("PERSON"), outputs=[mask_preview, mask_info])
        btn_face.click(lambda: visualize_semantic_mask("FACE"), outputs=[mask_preview, mask_info])
        btn_building.click(lambda: visualize_semantic_mask("BUILDING"), outputs=[mask_preview, mask_info])
        btn_vegetation.click(lambda: visualize_semantic_mask("VEGETATION"), outputs=[mask_preview, mask_info])
        btn_road.click(lambda: visualize_semantic_mask("ROAD"), outputs=[mask_preview, mask_info])
        btn_water.click(lambda: visualize_semantic_mask("WATER"), outputs=[mask_preview, mask_info])
        btn_others.click(lambda: visualize_semantic_mask("OTHERS"), outputs=[mask_preview, mask_info])
        
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
