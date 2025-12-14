"""
UI Layout - Gradio UI 布局定义

将 UI 布局从主文件分离，提高可维护性。
"""

import gradio as gr

from .theme import create_theme, get_css
from .config import STYLE_CHOICES, SEMANTIC_BUCKETS, build_ui_params
from .components import (
    create_all_region_ui_groups,
    collect_region_inputs,
    collect_realtime_region_inputs,
    create_mask_buttons,
    GlobalUIComponents,
)
from .state import ProcessingState
from .logic import full_compute, realtime_render, visualize_semantic_mask


def create_base_style_tab(global_comps: GlobalUIComponents):
    """创建基础风格 Tab 的内容"""
    gr.Markdown("### 1. 上传图片")
    input_image = gr.Image(label="上传图片", type="numpy", height=300)
    
    gr.Markdown("### 2. 风格化模式")
    with gr.Group():
        use_diffusion = gr.Checkbox(
            value=False,
            label="🎭 启用 AI 扩散风格化 (Diffusion)",
            info="启用后可生成 Shinkai/Hayao 等 AI 风格，但需要更长生成时间。"
        )
        gr.Markdown(
            "*💡 提示：若 Diffusion 模型未配置或加载失败，系统会自动降级为传统方法*"
        )
    
    gr.Markdown("### 3. 全局风格设置")
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
    
    gr.Markdown("### 4. 开始生成")
    process_btn = gr.Button(
        "✨ 生成卡通图像", 
        variant="primary", 
        elem_classes="generate-btn", 
        size="lg"
    )
    
    return input_image, use_diffusion, traditional_smooth_method, traditional_k, process_btn


def create_tune_tab(global_comps: GlobalUIComponents):
    """创建后期微调 Tab 的内容"""
    gr.Markdown("*以下参数调整可实时预览*")
    
    with gr.Accordion("🎨 色调与光影", open=True):
        global_comps.gamma = gr.Slider(0.5, 2.0, value=1.0, label="Gamma (明暗)", step=0.05)
        global_comps.saturation = gr.Slider(0.5, 1.5, value=1.0, label="饱和度 (鲜艳度)", step=0.05)
        global_comps.contrast = gr.Slider(0.5, 1.5, value=1.0, label="对比度", step=0.05)
        global_comps.brightness = gr.Slider(-50, 50, value=0, label="亮度微调")
    
    gr.Markdown("*线稿和细节增强参数已移至「区域精修」Tab，支持按语义区域分别设置*")
    
    # 隐藏的全局参数（保持兼容性）
    global_comps.edge_strength = gr.Slider(0, 1, value=0.5, visible=False)
    global_comps.line_engine = gr.Radio(["canny", "xdog"], value="canny", visible=False)
    global_comps.line_width = gr.Slider(0.5, 4, value=1, visible=False)
    global_comps.canny_low = gr.Slider(50, 150, value=100, visible=False)
    global_comps.canny_high = gr.Slider(100, 300, value=200, visible=False)
    global_comps.xdog_sigma = gr.Slider(0.1, 2.0, value=0.5, visible=False)
    global_comps.xdog_k = gr.Slider(1.0, 3.0, value=1.6, visible=False)
    global_comps.xdog_p = gr.Slider(5.0, 50.0, value=19.0, visible=False)
    global_comps.detail_enhance_enabled = gr.Checkbox(False, visible=False)
    global_comps.detail_strength = gr.Slider(0, 1, value=0.5, visible=False)


def create_advanced_tab(global_comps: GlobalUIComponents):
    """创建高级设置 Tab 的内容"""
    with gr.Group():
        gr.Markdown("**👤 人脸保护策略**")
        global_comps.face_protect_enabled = gr.Checkbox(True, label="启用人脸保护")
        global_comps.face_protect_mode = gr.Radio(
            ["protect", "blend", "full_style"], 
            value="protect", 
            label="模式"
        )
        global_comps.face_gan_weight_max = gr.Slider(0, 1, value=0.3, label="最大风格化权重")
    
    with gr.Group():
        gr.Markdown("**🎨 全局色彩协调**")
        global_comps.harmonization_enabled = gr.Checkbox(
            True, label="启用直方图匹配 (解决色调不一)"
        )
        global_comps.harmonization_reference = gr.Dropdown(
            SEMANTIC_BUCKETS + ["auto"], 
            value="SKY", 
            label="参考区域"
        )
        global_comps.harmonization_strength = gr.Slider(0, 1, value=0.8, label="匹配强度")

    with gr.Group():
        gr.Markdown("**🔀 融合算法**")
        global_comps.fusion_method = gr.Radio(
            ["soft_mask", "laplacian_pyramid", "poisson"], 
            value="soft_mask", 
            label="算法"
        )
        global_comps.fusion_blur_kernel = gr.Slider(5, 51, value=21, step=2, label="边缘模糊半径")


def create_ui():
    """
    创建 Gradio UI (Professional Version)
    
    Returns:
        gr.Blocks: Gradio 应用实例
    """
    theme = create_theme()
    css = get_css()
    
    with gr.Blocks(title="Catoon Pro - AI 图像风格化") as demo:
        # 初始化用户会话状态
        state = gr.State(ProcessingState())
        
        # 全局组件容器
        global_comps = GlobalUIComponents()
        
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
            # 左侧控制区
            with gr.Column(scale=1, min_width=350):
                with gr.Tabs():
                    # Tab 1: 基础风格
                    with gr.TabItem("🚀 基础风格", id="tab_base"):
                        (input_image, use_diffusion, traditional_smooth_method, 
                         traditional_k, process_btn) = create_base_style_tab(global_comps)
                    
                    # Tab 2: 后期微调
                    with gr.TabItem("🎛️ 后期微调", id="tab_tune"):
                        create_tune_tab(global_comps)
                    
                    # Tab 3: 区域精修
                    with gr.TabItem("🗺️ 区域精修", id="tab_region"):
                        gr.Markdown("### 指定特定区域的风格与后期效果")
                        gr.Markdown("*针对识别出的语义区域单独设置风格、线稿和细节增强*")
                        
                        with gr.Column(elem_classes="scroll-container"):
                            region_ui_map = create_all_region_ui_groups()
                    
                    # Tab 4: 高级设置
                    with gr.TabItem("⚙️ 高级", id="tab_adv"):
                        create_advanced_tab(global_comps)
            
            # 右侧预览区
            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="最终效果预览", 
                    type="numpy", 
                    elem_id="output_img", 
                    height=600
                )
                
                # 语义遮罩工具栏
                gr.Markdown("##### 🔍 语义层检视 (点击叠加显示)")
                mask_btns = create_mask_buttons()

                with gr.Accordion("遮罩调试视图", open=False, visible=True):
                    mask_preview = gr.Image(label="语义遮罩层", type="numpy", height=300)
                    mask_info = gr.Textbox(label="覆盖率信息", show_label=False)
        
        # 收集组件
        region_inputs = collect_region_inputs(region_ui_map)
        global_inputs = global_comps.get_all_components()
        realtime_region_inputs = collect_realtime_region_inputs(region_ui_map)
        
        # 事件处理函数
        def process_image(current_state, image, use_diff, smooth_method, k, *args):
            """完整处理（点击生成按钮时调用）"""
            if image is None:
                return None, current_state
            
            # Stage 1: 完整计算
            new_state = full_compute(
                current_state, image, smooth_method, int(k), use_diff
            )
            
            # 分离全局参数和区域参数
            n_global = len(global_inputs)
            global_args = args[:n_global]
            region_args = args[n_global:]
            
            # 构建 ui_params
            ui_params = build_ui_params(global_args, region_args)
            
            # Stage 2 + 3: 渲染
            result = realtime_render(new_state, ui_params)
            
            # 缓存渲染结果
            new_state.last_rendered_image = result
            
            return result, new_state
        
        def realtime_update(current_state, *args):
            """实时更新（参数变化时）"""
            if not current_state.is_ready():
                # 如果还没生成过，返回缓存的图像
                if current_state.last_rendered_image is not None:
                    return current_state.last_rendered_image, current_state
                return None, current_state
            
            # 分离全局参数和区域参数
            n_global = len(global_inputs)
            global_args = args[:n_global]
            region_args = args[n_global:]
            
            # 构建 ui_params
            ui_params = build_ui_params(global_args, region_args)
            
            # 计算稳定的参数哈希
            def make_hashable(obj):
                """递归将对象转换为可哈希的形式"""
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                elif isinstance(obj, (list, tuple)):
                    return tuple(make_hashable(x) for x in obj)
                elif isinstance(obj, float):
                    return round(obj, 4)  # 避免浮点精度问题
                else:
                    return obj
            
            args_hash = hash(make_hashable(ui_params))
            
            # 防止重复渲染
            if current_state.last_render_args_hash == args_hash:
                # 参数未变化，返回缓存的图像
                if current_state.last_rendered_image is not None:
                    return current_state.last_rendered_image, current_state
                return None, current_state
            
            new_state = current_state.copy()
            new_state.last_render_args_hash = args_hash
            
            # 渲染
            result = realtime_render(new_state, ui_params)
            
            # 缓存渲染结果
            new_state.last_rendered_image = result
            
            return result, new_state
        
        def show_mask(current_state, bucket):
            """显示语义遮罩"""
            img, info, new_state = visualize_semantic_mask(current_state, bucket)
            return img, info, new_state
        
        # 事件绑定
        all_inputs = [
            state, input_image, use_diffusion, traditional_smooth_method, traditional_k,
            *global_inputs, *region_inputs
        ]
        
        # 点击生成按钮
        process_btn.click(
            fn=process_image,
            inputs=all_inputs,
            outputs=[output_image, state]
        )
        
        # 实时更新组件列表
        realtime_inputs = [state, *global_inputs, *realtime_region_inputs]
        
        # 为所有实时组件绑定 change 事件
        all_realtime_components = list(global_inputs) + list(realtime_region_inputs)
        for component in all_realtime_components:
            if component is not None:
                component.change(
                    fn=realtime_update,
                    inputs=realtime_inputs,
                    outputs=[output_image, state]
                )
        
        # 遮罩按钮事件
        btn_map = mask_btns.get_all_buttons()
        for bucket, btn in btn_map.items():
            if btn is not None:
                btn.click(
                    fn=lambda s, b=bucket: show_mask(s, b),
                    inputs=[state],
                    outputs=[mask_preview, mask_info, state]
                )
    
    return {
        "demo": demo,
        "theme": theme,
        "css": css,
    }
