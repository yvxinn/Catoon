"""
UI State - 会话状态管理

使用 Gradio 的 gr.State 实现多用户并发支持。
每个用户独立维护自己的处理状态。
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ProcessingState:
    """
    存储单个用户的会话状态
    
    通过 gr.State 传递，实现多用户隔离。
    """
    
    # 原始图像
    original_image: np.ndarray | None = None
    image_hash: str | None = None
    
    # 中间结果缓存
    ctx: Any = None                    # 预处理上下文 (Context)
    seg_out: Any = None                # 分割结果 (SegmentationOutput)
    face_mask: np.ndarray | None = None
    candidates: dict | None = None     # 风格候选图 {style_id: StyleCandidate}
    
    # 渲染缓存（用于三级流水线）
    composited_image: np.ndarray | None = None  # 合成后的图像（Stage 2 输出）
    last_rendered_image: np.ndarray | None = None  # 最后渲染的图像（用于 Tab 切换时显示）
    
    # 参数缓存（用于判断是否需要重算）
    trad_params: tuple | None = None   # (k, smooth_method, use_diffusion)
    last_render_args_hash: str | None = None
    last_composite_args_hash: str | None = None
    
    # UI 状态
    active_masks: set = field(default_factory=set)  # 当前激活的语义遮罩
    use_diffusion: bool = False
    
    def is_ready(self) -> bool:
        """检查是否已完成初始计算，可以进行渲染"""
        return self.original_image is not None and self.candidates is not None
    
    def needs_inference(
        self, 
        image_hash: str,
        traditional_k: int,
        traditional_smooth_method: str,
        use_diffusion: bool
    ) -> bool:
        """判断是否需要重新推理（Stage 1）"""
        new_params = (traditional_k, traditional_smooth_method, use_diffusion)
        return self.image_hash != image_hash or self.trad_params != new_params
    
    def reset(self) -> None:
        """重置状态（上传新图片时）"""
        self.original_image = None
        self.image_hash = None
        self.ctx = None
        self.seg_out = None
        self.face_mask = None
        self.candidates = None
        self.composited_image = None
        self.last_rendered_image = None
        self.trad_params = None
        self.last_render_args_hash = None
        self.last_composite_args_hash = None
        self.active_masks = set()
        self.use_diffusion = False
    
    def copy(self) -> "ProcessingState":
        """创建状态副本（Gradio State 需要返回新对象才能触发更新）"""
        new_state = ProcessingState(
            original_image=self.original_image,
            image_hash=self.image_hash,
            ctx=self.ctx,
            seg_out=self.seg_out,
            face_mask=self.face_mask,
            candidates=self.candidates,
            composited_image=self.composited_image,
            last_rendered_image=self.last_rendered_image,
            trad_params=self.trad_params,
            last_render_args_hash=self.last_render_args_hash,
            last_composite_args_hash=self.last_composite_args_hash,
            active_masks=set(self.active_masks),
            use_diffusion=self.use_diffusion,
        )
        return new_state


def compute_image_hash(image: np.ndarray) -> str:
    """计算图像哈希（用于缓存判断）"""
    return str(hash(image.tobytes()))

