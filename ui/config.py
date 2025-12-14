"""
UI Config - 参数数据类定义

将 UI 参数组织为结构化的数据类，消除"参数爆炸"问题。
"""

from dataclasses import dataclass, field


# 语义区域常量

SEMANTIC_BUCKETS = ["SKY", "PERSON", "BUILDING", "VEGETATION", "ROAD", "WATER", "OTHERS"]

STYLE_CHOICES = ["Traditional", "Hayao", "Shinkai", "Paprika"]

# 语义区域颜色映射（用于遮罩可视化）
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


# 参数数据类

@dataclass
class LineartParams:
    """线稿引擎参数"""
    engine: str = "canny"        # canny | xdog
    width: float = 1.0           # 线条粗细
    canny_low: int = 100         # Canny 低阈值
    canny_high: int = 200        # Canny 高阈值
    xdog_sigma: float = 0.5      # XDoG Sigma
    xdog_k: float = 1.6          # XDoG K
    xdog_p: float = 19.0         # XDoG P


@dataclass
class RegionParams:
    """单个区域（如天空、人物）的参数集"""
    style: str = "Traditional"
    strength: float = 1.0
    k: int = 16                   # KMeans K 值
    lineart_strength: float = 0.5
    detail_enhance: float = 0.0
    
    # 线稿引擎参数
    line_engine: str = "canny"
    line_width: float = 1.0
    canny_low: int = 100
    canny_high: int = 200
    xdog_sigma: float = 0.5
    xdog_k: float = 1.6
    xdog_p: float = 19.0
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于传递给 pipeline）"""
        return {
            "style": self.style,
            "strength": self.strength,
            "k": self.k,
            "lineart_strength": self.lineart_strength,
            "detail_enhance": self.detail_enhance,
            "line_engine": self.line_engine,
            "line_width": self.line_width,
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "xdog_sigma": self.xdog_sigma,
            "xdog_k": self.xdog_k,
            "xdog_p": self.xdog_p,
        }


@dataclass
class GlobalParams:
    """全局渲染参数"""
    
    # 融合
    fusion_method: str = "soft_mask"
    fusion_blur_kernel: int = 21
    
    # 协调
    harmonization_enabled: bool = True
    harmonization_reference: str = "SKY"
    harmonization_strength: float = 0.8
    
    # 色调
    gamma: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    brightness: float = 0.0
    
    # 人脸保护
    face_protect_enabled: bool = True
    face_protect_mode: str = "protect"
    face_gan_weight_max: float = 0.3
    
    # 全局线稿（作为默认值，已弃用但保持兼容）
    edge_strength: float = 0.5
    line_engine: str = "canny"
    line_width: int = 1
    canny_low: int = 100
    canny_high: int = 200
    xdog_sigma: float = 0.5
    xdog_k: float = 1.6
    xdog_p: float = 19.0
    detail_enhance_enabled: bool = False
    detail_strength: float = 0.5


# 区域默认配置

# 默认线稿参数（所有区域共用）
_DEFAULT_LINEART = {
    "line_engine": "canny",
    "line_width": 1.0,
    "canny_low": 100,
    "canny_high": 200,
    "xdog_sigma": 0.5,
    "xdog_k": 1.6,
    "xdog_p": 19.0,
}

def _make_region(style: str, strength: float, k: int, lineart_strength: float, detail_enhance: float) -> dict:
    """生成区域配置"""
    return {
        "style": style, "strength": strength, "k": k,
        "lineart_strength": lineart_strength, "detail_enhance": detail_enhance,
        **_DEFAULT_LINEART
    }

REGION_DEFAULTS: dict[str, dict] = {
    "SKY":        _make_region("Shinkai", 1.0, 16, 0.3, 0.0),
    "PERSON":     _make_region("Traditional", 0.7, 20, 0.6, 0.3),
    "BUILDING":   _make_region("Traditional", 1.0, 16, 0.7, 0.2),
    "VEGETATION": _make_region("Hayao", 1.0, 24, 0.4, 0.5),
    "ROAD":       _make_region("Traditional", 1.0, 12, 0.5, 0.1),
    "WATER":      _make_region("Shinkai", 1.0, 16, 0.2, 0.0),
    "OTHERS":     _make_region("Traditional", 1.0, 16, 0.5, 0.2),
}

# 区域显示配置（用于 UI 生成）
REGION_UI_CONFIG = {
    "SKY": {"label": "☁️ 天空", "emoji": "☁️"},
    "PERSON": {"label": "👤 人物", "emoji": "👤"},
    "BUILDING": {"label": "🏠 建筑", "emoji": "🏠"},
    "VEGETATION": {"label": "🌳 植被", "emoji": "🌳"},
    "ROAD": {"label": "🛤️ 道路", "emoji": "🛤️"},
    "WATER": {"label": "🌊 水体", "emoji": "🌊"},
    "OTHERS": {"label": "📦 其他", "emoji": "📦"},
}


# 参数解析辅助函数

def parse_region_params_from_flat_args(
    bucket: str,
    style: str,
    strength: float,
    k: int,
    lineart: float,
    detail: float,
    line_engine: str,
    line_width: float,
    canny_low: int,
    canny_high: int,
    xdog_sigma: float,
    xdog_k: float,
    xdog_p: float,
) -> RegionParams:
    """从扁平参数构建 RegionParams 对象"""
    return RegionParams(
        style=style,
        strength=strength,
        k=int(k),
        lineart_strength=lineart,
        detail_enhance=detail,
        line_engine=line_engine,
        line_width=line_width,
        canny_low=int(canny_low),
        canny_high=int(canny_high),
        xdog_sigma=xdog_sigma,
        xdog_k=xdog_k,
        xdog_p=xdog_p,
    )


def build_region_overrides(region_params_map: dict[str, RegionParams]) -> dict[str, dict]:
    """将 RegionParams 字典转换为 UI 参数格式"""
    return {bucket: params.to_dict() for bucket, params in region_params_map.items()}


# 每个区域的参数数量（用于解析扁平参数列表）
PARAMS_PER_REGION = 12  # style, strength, k, lineart, detail, line_engine, line_width, canny_low, canny_high, xdog_sigma, xdog_k, xdog_p


def parse_flat_region_args(flat_args: tuple) -> dict[str, dict]:
    """
    将扁平的区域参数列表解析为结构化字典
    
    Args:
        flat_args: 按 SEMANTIC_BUCKETS 顺序排列的扁平参数元组
    
    Returns:
        {bucket: {param_name: value, ...}, ...}
    """
    region_overrides = {}
    
    for i, bucket in enumerate(SEMANTIC_BUCKETS):
        start = i * PARAMS_PER_REGION
        args = flat_args[start:start + PARAMS_PER_REGION]
        
        if len(args) < PARAMS_PER_REGION:
            continue
        
        (style, strength, k, lineart, detail,
         line_engine, line_width, canny_low, canny_high,
         xdog_sigma, xdog_k, xdog_p) = args
        
        region_overrides[bucket] = {
            "style": style,
            "strength": strength,
            "k": int(k),
            "lineart_strength": lineart,
            "detail_enhance": detail,
            "line_engine": line_engine,
            "line_width": line_width,
            "canny_low": int(canny_low),
            "canny_high": int(canny_high),
            "xdog_sigma": xdog_sigma,
            "xdog_k": xdog_k,
            "xdog_p": xdog_p,
        }
    
    return region_overrides


def build_ui_params(
    global_args: tuple,
    region_args: tuple
) -> dict:
    """
    构建完整的 ui_params 字典
    
    Args:
        global_args: 全局参数元组（按 GlobalUIComponents.get_all_components 顺序）
        region_args: 区域参数元组（按 SEMANTIC_BUCKETS 顺序）
    
    Returns:
        完整的 ui_params 字典
    """
    # 解析全局参数（按 GlobalUIComponents.get_all_components 顺序）
    (fusion_method, fusion_blur_kernel,
     harmonization_enabled, harmonization_reference, harmonization_strength,
     edge_strength, line_engine, line_width,
     canny_low, canny_high, xdog_sigma, xdog_k, xdog_p,
     detail_enhance_enabled, detail_strength,
     gamma, contrast, saturation, brightness,
     face_protect_enabled, face_protect_mode, face_gan_weight_max) = global_args
    
    # 解析区域参数
    region_overrides = parse_flat_region_args(region_args)
    
    return {
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
        "region_overrides": region_overrides,
    }
