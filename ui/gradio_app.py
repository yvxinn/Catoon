"""
Gradio UI - 交互式卡通化界面 (Professional Version)

面向客户的现代化界面，提供可视化的参数调整和实时预览。
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
    """创建 Gradio UI (Professional Version)"""
    
    style_choices = ["Traditional", "Hayao", "Shinkai", "Paprika"]
    semantic_buckets = ["SKY", "PERSON", "BUILDING", "VEGETATION", "ROAD", "WATER", "OTHERS"]
    
    # 定制主题 - 使用更专业的蓝紫色调
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_md,
    )

    # 自定义 CSS：增加滚动容器样式
    # 重要修正：.scroll-container 使用 display: block !important 防止 Flex 压缩子元素
    css = """
    .gradio-container {
        font-family: 'Helvetica Neue', 'Segoe UI', Roboto, sans-serif;
    }
    .generate-btn {
        background: linear-gradient(90deg, #6366f1 0%, #4338ca 100%) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.1s;
    }
    .generate-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .mask-btn {
        font-size: 0.8rem !important;
        padding: 4px 8px !important;
    }
    /* 核心修改：内部滚动容器 */
    .scroll-container {
        max-height: 650px;       /* 限制最大高度 */
        overflow-y: auto;        /* 允许垂直滚动 */
        padding-right: 12px;     /* 给滚动条留出空间 */
        border-radius: 8px;
        background-color: rgba(249, 250, 251, 0.5); /* 极淡的背景色区分 */
        display: block !important; /* 【关键】强制块级布局，防止 Gradio 的 flex 压缩子元素 */
    }
    /* 手动补充子元素间距 (因为 block 布局不支持 gap) */
    .scroll-container > * {
        margin-bottom: 16px;
    }
    .scroll-container > *:last-child {
        margin-bottom: 0;
    }
    
    /* 美化滚动条 */
    .scroll-container::-webkit-scrollbar {
        width: 6px;
    }
    .scroll-container::-webkit-scrollbar-thumb {
        background-color: #cbd5e1;
        border-radius: 4px;
    }
    .scroll-container::-webkit-scrollbar-track {
        background-color: transparent;
    }
    """

    with gr.Blocks(title="Catoon Pro - AI 图像风格化", theme=theme, css=css) as demo:
        
        # 顶栏
        with gr.Row(elem_classes="header"):
            with gr.Column():
                gr.Markdown(
                    """
                    # 🎨 Catoon Pro
                    ### 语义感知可控卡通化工作站
                    """
                )
        
        with gr.Row():
            # ================== 左侧控制区 (Tabbed) ==================
            with gr.Column(scale=1, min_width=350):
                
                with gr.Tabs():
                    
                    # Tab 1: 基础风格 (Base Style) - 用户入口
                    with gr.TabItem("🚀 基础风格", id="tab_base"):
                        gr.Markdown("### 1. 上传图片与选择基础模式")
                        input_image = gr.Image(label="上传图片", type="numpy", height=300)
                        
                        gr.Markdown("### 2. 全局风格设置")
                        with gr.Group():
                            traditional_smooth_method = gr.Dropdown(
                                choices=["bilateral", "edge_preserving", "mean_shift"],
                                value="bilateral",
                                label="平滑算法",
                                info="决定画面的'色块感'程度"
                            )
                            traditional_k = gr.Slider(
                                4, 48, value=16, step=4,
                                label="色彩量化 (K值)",
                                info="数值越小，颜色越简化，卡通感越强"
                            )
                        
                        gr.Markdown("### 3. 开始生成")
                        process_btn = gr.Button("✨ 生成卡通图像", variant="primary", elem_classes="generate-btn", size="lg")

                    # Tab 2: 后期微调 (Fine-tuning) - 实时调整
                    with gr.TabItem("🎛️ 后期微调", id="tab_tune"):
                        gr.Markdown("*以下参数调整可实时预览*")
                        
                        with gr.Accordion("🎨 色调与光影", open=True):
                            gamma = gr.Slider(0.5, 2.0, value=1.0, label="Gamma (明暗)", step=0.05)
                            saturation = gr.Slider(0.5, 1.5, value=1.0, label="饱和度 (鲜艳度)", step=0.05)
                            contrast = gr.Slider(0.5, 1.5, value=1.0, label="对比度", step=0.05)
                            brightness = gr.Slider(-50, 50, value=0, label="亮度微调")

                        with gr.Accordion("✏️ 线稿增强", open=True):
                            edge_strength = gr.Slider(0, 1, value=0.5, label="线稿不透明度")
                            line_engine = gr.Radio(["canny", "xdog"], value="canny", label="引擎", interactive=True)
                            line_width = gr.Slider(0.5, 4, value=1, step=0.25, label="线条粗细")
                            
                            with gr.Group(visible=True):
                                canny_low = gr.Slider(50, 150, value=100, label="Canny 低阈值")
                                canny_high = gr.Slider(100, 300, value=200, label="Canny 高阈值")
                                xdog_sigma = gr.Slider(0.1, 2.0, value=0.5, label="XDoG Sigma")
                                xdog_k = gr.Slider(1.0, 3.0, value=1.6, label="XDoG K")
                                xdog_p = gr.Slider(5.0, 50.0, value=19.0, label="XDoG P")

                        with gr.Accordion("🔍 纹理细节", open=False):
                            detail_enhance_enabled = gr.Checkbox(False, label="启用纹理增强 (Guided Filter)")
                            detail_strength = gr.Slider(0, 1, value=0.5, label="纹理强度")

                    # Tab 3: 区域精修 (Region Styles) - 核心修改部分
                    with gr.TabItem("🗺️ 区域精修", id="tab_region"):
                        gr.Markdown("### 指定特定区域的风格")
                        gr.Markdown("*针对识别出的语义区域单独设置风格*")
                        
                        # 使用 scroll-container 包裹所有区域设置，并取消折叠
                        # CSS 中已设置 display: block !important 避免布局崩坏
                        with gr.Column(elem_classes="scroll-container"):
                            
                            with gr.Group():
                                sky_style = gr.Dropdown(style_choices, value="Shinkai", label="☁️ 天空")
                                sky_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                sky_k = gr.Slider(4, 64, value=16, step=2, label="K值 (Traditional)", visible=True) 

                            with gr.Group():
                                person_style = gr.Dropdown(style_choices, value="Traditional", label="👤 人物")
                                person_strength = gr.Slider(0, 1, value=0.7, label="强度")
                                person_k = gr.Slider(4, 64, value=20, step=2, label="K值 (Traditional)",visible=True)

                            with gr.Group():
                                building_style = gr.Dropdown(style_choices, value="Traditional", label="🏠 建筑")
                                building_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                building_k = gr.Slider(4, 64, value=16, step=2, label="K值 (Traditional)",visible=True)

                            with gr.Group():
                                vegetation_style = gr.Dropdown(style_choices, value="Hayao", label="🌳 植被")
                                vegetation_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                vegetation_k = gr.Slider(4, 64, value=24, step=2, label="K值 (Traditional)",visible=True)

                            # 移除了 Accordion，直接平铺显示
                            with gr.Group():
                                road_style = gr.Dropdown(style_choices, value="Traditional", label="🛤️ 道路")
                                road_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                road_k = gr.Slider(4, 64, value=12, step=2, label="K值 (Traditional)",visible=True)
                                
                            with gr.Group():
                                water_style = gr.Dropdown(style_choices, value="Shinkai", label="🌊 水体")
                                water_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                water_k = gr.Slider(4, 64, value=16, step=2, label="K值 (Traditional)",visible=True)
                                
                            with gr.Group():
                                others_style = gr.Dropdown(style_choices, value="Traditional", label="📦 其他")
                                others_strength = gr.Slider(0, 1, value=1.0, label="强度")
                                others_k = gr.Slider(4, 64, value=16, step=2, label="K值 (Traditional)",visible=True)

                    # Tab 4: 高级设置 (Advanced)
                    with gr.TabItem("⚙️ 高级", id="tab_adv"):
                        
                        with gr.Group():
                            gr.Markdown("**👤 人脸保护策略**")
                            face_protect_enabled = gr.Checkbox(True, label="启用人脸保护")
                            face_protect_mode = gr.Radio(["protect", "blend", "full_style"], value="protect", label="模式")
                            face_gan_weight_max = gr.Slider(0, 1, value=0.3, label="最大风格化权重")
                        
                        with gr.Group():
                            gr.Markdown("**🎨 全局色彩协调**")
                            harmonization_enabled = gr.Checkbox(True, label="启用直方图匹配 (解决色调不一)")
                            harmonization_reference = gr.Dropdown(semantic_buckets + ["auto"], value="SKY", label="参考区域")
                            harmonization_strength = gr.Slider(0, 1, value=0.8, label="匹配强度")

                        with gr.Group():
                            gr.Markdown("**🔀 融合算法**")
                            fusion_method = gr.Radio(["soft_mask", "laplacian_pyramid", "poisson"], value="soft_mask", label="算法")
                            fusion_blur_kernel = gr.Slider(5, 51, value=21, step=2, label="边缘模糊半径")
            
            # ================== 右侧预览区 ==================
            with gr.Column(scale=2):
                output_image = gr.Image(label="最终效果预览", type="numpy", elem_id="output_img", height=600)
                
                # 语义遮罩工具栏
                gr.Markdown("##### 🔍 语义层检视 (点击叠加显示)")
                with gr.Row(elem_id="mask_toolbar"):
                    btn_none = gr.Button("🔄 原图", size="sm", elem_classes="mask-btn")
                    btn_sky = gr.Button("☁️ 天空", size="sm", elem_classes="mask-btn")
                    btn_person = gr.Button("👤 人物", size="sm", elem_classes="mask-btn")
                    btn_face = gr.Button("😊 面部", size="sm", elem_classes="mask-btn")
                    btn_building = gr.Button("🏠 建筑", size="sm", elem_classes="mask-btn")
                    btn_vegetation = gr.Button("🌳 植被", size="sm", elem_classes="mask-btn")
                    btn_road = gr.Button("🛤️ 道路", size="sm", elem_classes="mask-btn")
                    btn_water = gr.Button("🌊 水体", size="sm", elem_classes="mask-btn")
                    btn_others = gr.Button("📦 其他", size="sm", elem_classes="mask-btn")

                with gr.Accordion("遮罩调试视图", open=False, visible=True):
                    mask_preview = gr.Image(label="语义遮罩层", type="numpy", height=300)
                    mask_info = gr.Textbox(label="覆盖率信息", show_label=False)

        # 整理所有输入
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
        
        # 实时调整参数列表
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
        
        # ================== 事件绑定 ==================
        process_btn.click(
            fn=process_image,
            inputs=all_inputs,
            outputs=output_image
        )

        # 上传图片后自动处理，保持原有“即传即算”体验
        input_image.change(
            fn=process_image,
            inputs=all_inputs,
            outputs=output_image
        )
        
        def realtime_update(*args):
            """实时更新（仅当缓存存在时）"""
            if _cache["candidates"] is None:
                return None 
            return realtime_render(*args)
        
        for component in realtime_components:
            component.change(
                fn=realtime_update,
                inputs=realtime_components,
                outputs=output_image
            )
        
        btn_none.click(lambda: visualize_semantic_mask("NONE"), outputs=[mask_preview, mask_info])
        btn_sky.click(lambda: visualize_semantic_mask("SKY"), outputs=[mask_preview, mask_info])
        btn_person.click(lambda: visualize_semantic_mask("PERSON"), outputs=[mask_preview, mask_info])
        btn_face.click(lambda: visualize_semantic_mask("FACE"), outputs=[mask_preview, mask_info])
        btn_building.click(lambda: visualize_semantic_mask("BUILDING"), outputs=[mask_preview, mask_info])
        btn_vegetation.click(lambda: visualize_semantic_mask("VEGETATION"), outputs=[mask_preview, mask_info])
        btn_road.click(lambda: visualize_semantic_mask("ROAD"), outputs=[mask_preview, mask_info])
        btn_water.click(lambda: visualize_semantic_mask("WATER"), outputs=[mask_preview, mask_info])
        btn_others.click(lambda: visualize_semantic_mask("OTHERS"), outputs=[mask_preview, mask_info])
        
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )