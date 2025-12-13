"""
UI Components - UI 组件工厂函数

消除重复代码，通过工厂函数生成区域 UI 组件。
"""

import gradio as gr
from dataclasses import dataclass, field
from typing import Any

from .config import STYLE_CHOICES, REGION_DEFAULTS, REGION_UI_CONFIG, SEMANTIC_BUCKETS


@dataclass
class RegionUIComponents:
    """单个区域的 UI 组件引用集合"""
    bucket: str
    style: gr.Dropdown = None
    strength: gr.Slider = None
    k: gr.Slider = None
    lineart: gr.Slider = None
    detail: gr.Slider = None
    line_engine: gr.Radio = None
    line_width: gr.Slider = None
    canny_low: gr.Slider = None
    canny_high: gr.Slider = None
    xdog_sigma: gr.Slider = None
    xdog_k: gr.Slider = None
    xdog_p: gr.Slider = None
    
    def get_all_components(self) -> list:
        """返回所有组件的有序列表（用于 Gradio inputs）"""
        return [
            self.style, self.strength, self.k, self.lineart, self.detail,
            self.line_engine, self.line_width, 
            self.canny_low, self.canny_high,
            self.xdog_sigma, self.xdog_k, self.xdog_p,
        ]
    
    def get_realtime_components(self) -> list:
        """返回需要实时更新的组件列表"""
        return self.get_all_components()  # 所有区域参数都支持实时更新


def create_region_ui_group(bucket: str) -> RegionUIComponents:
    """
    生成单个区域的 UI 组件组
    
    Args:
        bucket: 语义桶名称 (SKY, PERSON, etc.)
    
    Returns:
        RegionUIComponents: 包含所有组件引用的对象
    """
    defaults = REGION_DEFAULTS.get(bucket, REGION_DEFAULTS["OTHERS"])
    ui_config = REGION_UI_CONFIG.get(bucket, {"label": f"📦 {bucket}", "emoji": "📦"})
    
    components = RegionUIComponents(bucket=bucket)
    
    with gr.Group():
        gr.Markdown(f"##### {ui_config['label']}")
        
        components.style = gr.Dropdown(
            choices=STYLE_CHOICES,
            value=defaults["style"],
            label="风格"
        )
        components.strength = gr.Slider(
            0, 1, 
            value=defaults["strength"],
            label="风格强度"
        )
        components.k = gr.Slider(
            4, 64, 
            value=defaults["k"],
            step=2,
            label="K值 (Traditional)"
        )
        
        with gr.Accordion("✏️ 线稿参数", open=False):
            components.lineart = gr.Slider(
                0, 1,
                value=defaults["lineart_strength"],
                label="线稿强度"
            )
            components.line_engine = gr.Radio(
                ["canny", "xdog"],
                value=defaults["line_engine"],
                label="引擎"
            )
            components.line_width = gr.Slider(
                0.5, 4,
                value=defaults["line_width"],
                step=0.25,
                label="线条粗细"
            )
            components.canny_low = gr.Slider(
                50, 150,
                value=defaults["canny_low"],
                label="Canny 低阈值"
            )
            components.canny_high = gr.Slider(
                100, 300,
                value=defaults["canny_high"],
                label="Canny 高阈值"
            )
            components.xdog_sigma = gr.Slider(
                0.1, 2.0,
                value=defaults["xdog_sigma"],
                label="XDoG Sigma"
            )
            components.xdog_k = gr.Slider(
                1.0, 3.0,
                value=defaults["xdog_k"],
                label="XDoG K"
            )
            components.xdog_p = gr.Slider(
                5.0, 50.0,
                value=defaults["xdog_p"],
                label="XDoG P"
            )
            components.detail = gr.Slider(
                0, 1,
                value=defaults["detail_enhance"],
                label="🔍 细节增强"
            )
    
    return components


def create_all_region_ui_groups() -> dict[str, RegionUIComponents]:
    """
    创建所有语义区域的 UI 组件
    
    Returns:
        dict: {bucket_name: RegionUIComponents}
    """
    region_ui_map = {}
    for bucket in SEMANTIC_BUCKETS:
        region_ui_map[bucket] = create_region_ui_group(bucket)
    return region_ui_map


def collect_region_inputs(region_ui_map: dict[str, RegionUIComponents]) -> list:
    """
    收集所有区域组件为扁平列表（用于 Gradio inputs）
    
    顺序：按 SEMANTIC_BUCKETS 顺序，每个区域内按 get_all_components() 顺序
    """
    inputs = []
    for bucket in SEMANTIC_BUCKETS:
        if bucket in region_ui_map:
            inputs.extend(region_ui_map[bucket].get_all_components())
    return inputs


def collect_realtime_region_inputs(region_ui_map: dict[str, RegionUIComponents]) -> list:
    """收集需要实时更新的区域组件"""
    inputs = []
    for bucket in SEMANTIC_BUCKETS:
        if bucket in region_ui_map:
            inputs.extend(region_ui_map[bucket].get_realtime_components())
    return inputs


# ============== 全局参数组件 ==============

@dataclass
class GlobalUIComponents:
    """全局参数 UI 组件引用"""
    # 融合
    fusion_method: gr.Radio = None
    fusion_blur_kernel: gr.Slider = None
    
    # 协调
    harmonization_enabled: gr.Checkbox = None
    harmonization_reference: gr.Dropdown = None
    harmonization_strength: gr.Slider = None
    
    # 色调
    gamma: gr.Slider = None
    contrast: gr.Slider = None
    saturation: gr.Slider = None
    brightness: gr.Slider = None
    
    # 人脸保护
    face_protect_enabled: gr.Checkbox = None
    face_protect_mode: gr.Radio = None
    face_gan_weight_max: gr.Slider = None
    
    # 全局线稿（隐藏，保持兼容）
    edge_strength: gr.Slider = None
    line_engine: gr.Radio = None
    line_width: gr.Slider = None
    canny_low: gr.Slider = None
    canny_high: gr.Slider = None
    xdog_sigma: gr.Slider = None
    xdog_k: gr.Slider = None
    xdog_p: gr.Slider = None
    detail_enhance_enabled: gr.Checkbox = None
    detail_strength: gr.Slider = None
    
    def get_all_components(self) -> list:
        """返回所有组件的有序列表"""
        return [
            self.fusion_method, self.fusion_blur_kernel,
            self.harmonization_enabled, self.harmonization_reference, self.harmonization_strength,
            self.edge_strength, self.line_engine, self.line_width,
            self.canny_low, self.canny_high, self.xdog_sigma, self.xdog_k, self.xdog_p,
            self.detail_enhance_enabled, self.detail_strength,
            self.gamma, self.contrast, self.saturation, self.brightness,
            self.face_protect_enabled, self.face_protect_mode, self.face_gan_weight_max,
        ]
    
    def get_realtime_components(self) -> list:
        """返回需要实时更新的组件"""
        return self.get_all_components()


def create_global_ui_components() -> GlobalUIComponents:
    """创建全局参数 UI 组件（在各自的 Tab 中调用）"""
    return GlobalUIComponents()


# ============== 遮罩按钮组件 ==============

@dataclass
class MaskButtonComponents:
    """遮罩可视化按钮组件"""
    btn_none: gr.Button = None
    btn_sky: gr.Button = None
    btn_person: gr.Button = None
    btn_face: gr.Button = None
    btn_building: gr.Button = None
    btn_vegetation: gr.Button = None
    btn_road: gr.Button = None
    btn_water: gr.Button = None
    btn_others: gr.Button = None
    
    def get_all_buttons(self) -> dict[str, gr.Button]:
        """返回所有按钮的字典"""
        return {
            "NONE": self.btn_none,
            "SKY": self.btn_sky,
            "PERSON": self.btn_person,
            "FACE": self.btn_face,
            "BUILDING": self.btn_building,
            "VEGETATION": self.btn_vegetation,
            "ROAD": self.btn_road,
            "WATER": self.btn_water,
            "OTHERS": self.btn_others,
        }


def create_mask_buttons() -> MaskButtonComponents:
    """创建遮罩可视化按钮组"""
    components = MaskButtonComponents()
    
    with gr.Row(elem_id="mask_toolbar"):
        components.btn_none = gr.Button("🔄 原图", size="sm", elem_classes="mask-btn")
        components.btn_sky = gr.Button("☁️ 天空", size="sm", elem_classes="mask-btn")
        components.btn_person = gr.Button("👤 人物", size="sm", elem_classes="mask-btn")
        components.btn_face = gr.Button("😊 面部", size="sm", elem_classes="mask-btn")
        components.btn_building = gr.Button("🏠 建筑", size="sm", elem_classes="mask-btn")
        components.btn_vegetation = gr.Button("🌳 植被", size="sm", elem_classes="mask-btn")
        components.btn_road = gr.Button("🛤️ 道路", size="sm", elem_classes="mask-btn")
        components.btn_water = gr.Button("🌊 水体", size="sm", elem_classes="mask-btn")
        components.btn_others = gr.Button("📦 其他", size="sm", elem_classes="mask-btn")
    
    return components
