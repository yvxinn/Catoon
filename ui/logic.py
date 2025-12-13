"""
UI Logic - 核心业务逻辑

将处理逻辑从 UI 层分离，支持多用户并发。
"""

import numpy as np
from typing import Any

from .state import ProcessingState, compute_image_hash
from .config import SEMANTIC_COLORS, SEMANTIC_BUCKETS

# Pipeline 延迟导入
_pipeline = None


def get_pipeline():
    """懒加载 Pipeline"""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import load_pipeline
        _pipeline = load_pipeline()
    return _pipeline


def full_compute(
    state: ProcessingState,
    image: np.ndarray,
    traditional_smooth_method: str,
    traditional_k: int,
    use_diffusion: bool = False
) -> ProcessingState:
    """
    完整计算（需要模型推理）- Stage 1
    
    Args:
        state: 用户会话状态
        image: 输入图像
        traditional_smooth_method: 平滑方法
        traditional_k: K 值
        use_diffusion: 是否启用 Diffusion
    
    Returns:
        更新后的状态
    """
    if image is None:
        return state
    
    pipe = get_pipeline()
    img_hash = compute_image_hash(image)
    trad_params = (traditional_k, traditional_smooth_method, use_diffusion)
    
    # 检查是否需要重新计算
    if state.image_hash == img_hash and state.trad_params == trad_params:
        return state  # 使用缓存
    
    print("[Pipeline] 执行完整计算...")
    print(f"[Pipeline] Diffusion 模式: {'启用' if use_diffusion else '关闭'}")
    
    # 创建新状态
    new_state = state.copy()
    
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
        "traditional_smooth_method": traditional_smooth_method,
        "use_diffusion": use_diffusion
    }
    
    # 先生成边缘图和传统风格候选
    edge_map = pipe._get_or_build_edge_map(ctx, ui_params)
    trad_candidate = pipe._get_or_build_traditional(ctx, ui_params)
    
    # 根据 use_diffusion 决定是否生成 Diffusion 候选
    if use_diffusion:
        try:
            candidates = pipe._get_or_build_candidates(ctx, edge_map, trad_candidate, ui_params)
            print("[Pipeline] Diffusion 候选生成完成")
        except Exception as e:
            print(f"[Pipeline] Diffusion 生成失败，降级为传统方法: {e}")
            candidates = {"Traditional": trad_candidate}
    else:
        candidates = {"Traditional": trad_candidate}
        print("[Pipeline] 使用传统方法（Diffusion 已关闭）")
    
    # 更新状态
    new_state.image_hash = img_hash
    new_state.ctx = ctx
    new_state.seg_out = seg_out
    new_state.face_mask = face_mask
    new_state.candidates = candidates
    new_state.trad_params = trad_params
    new_state.original_image = image.copy()
    new_state.active_masks = set()
    new_state.use_diffusion = use_diffusion
    
    print("[Pipeline] 完整计算完成，已缓存中间结果")
    return new_state


def apply_tone_adjustment(
    image: np.ndarray,
    gamma: float,
    contrast: float,
    saturation: float,
    brightness: float
) -> np.ndarray:
    """应用色调调整 - Stage 3（最轻量）"""
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
        result_u8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(result_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = result_u8.astype(np.float32) / 255.0
    
    return np.clip(result, 0, 1).astype(np.float32)


def _unify_sizes(
    candidates: dict,
    seg_out,
    face_mask: np.ndarray | None,
    original_image: np.ndarray
) -> tuple[dict, dict, np.ndarray | None, np.ndarray]:
    """
    统一所有图像和 mask 的尺寸
    
    以第一个候选图像的尺寸为基准，调整所有其他数据的尺寸。
    
    Args:
        candidates: 风格候选字典
        seg_out: 分割输出
        face_mask: 人脸遮罩
        original_image: 原图
    
    Returns:
        (candidates, semantic_masks, face_mask, original_image) - 尺寸统一后的数据
    """
    import cv2
    
    # 获取目标尺寸（以第一个候选图像为基准）
    first_candidate = next(iter(candidates.values()))
    target_h, target_w = first_candidate.image.shape[:2]
    
    # 调整 semantic_masks
    unified_masks = {}
    for bucket_name, mask in seg_out.semantic_masks.items():
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        unified_masks[bucket_name] = mask
    
    # 调整 face_mask
    unified_face_mask = face_mask
    if face_mask is not None and face_mask.shape[:2] != (target_h, target_w):
        unified_face_mask = cv2.resize(face_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 调整 original_image
    unified_original = original_image
    if original_image.shape[:2] != (target_h, target_w):
        unified_original = cv2.resize(original_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    return candidates, unified_masks, unified_face_mask, unified_original


def realtime_render(
    state: ProcessingState,
    ui_params: dict
) -> np.ndarray | None:
    """
    实时渲染（不重新推理）- Stage 2 + Stage 3
    
    Args:
        state: 用户会话状态（包含缓存的中间结果）
        ui_params: UI 参数字典
    
    Returns:
        渲染后的图像 (uint8)
    """
    if not state.is_ready():
        return None
    
    pipe = get_pipeline()
    ctx = state.ctx
    seg_out = state.seg_out
    face_mask = state.face_mask
    candidates = state.candidates
    
    # 统一所有数据的尺寸（解决分割输出与风格化图像尺寸不匹配的问题）
    candidates, unified_masks, face_mask, original_image = _unify_sizes(
        candidates, seg_out, face_mask, ctx.image_f32
    )
    
    # 创建统一尺寸后的 seg_out 副本
    from src.context import SegmentationOutput
    unified_seg_out = SegmentationOutput(
        label_map=seg_out.label_map,  # label_map 不需要在融合中使用
        semantic_masks=unified_masks,
        seg_logits=seg_out.seg_logits
    )
    
    # D. 语义路由（轻量）
    routing = pipe.router.route(
        semantic_masks=unified_seg_out.semantic_masks,
        face_mask=face_mask,
        ui_overrides=ui_params
    )
    
    # C2. 区域级风格化（按需生成，带缓存）
    region_candidates = pipe.region_stylizer.generate_region_styles(
        image_f32=original_image,
        image_hash=ctx.image_hash,
        seg_out=unified_seg_out,
        region_configs=routing.region_configs,
        global_candidates=candidates
    )
    
    # E. 区域融合（轻量）
    fused = pipe.fuser.fuse(
        candidates=candidates,
        routing=routing,
        seg_out=unified_seg_out,
        method=ui_params.get("fusion_method", "soft_mask"),
        blur_kernel=ui_params.get("fusion_blur_kernel", 21),
        original_image=original_image,
        region_candidates=region_candidates
    )
    
    # F. 全局协调（轻量）
    if ui_params.get("harmonization_enabled", True):
        ref = pipe.harmonizer.pick_reference(
            candidates, unified_seg_out, ui_params, pipe.cfg.harmonization
        )
        fused = pipe.harmonizer.match_and_adjust(fused, ref, ui_params)
    
    # G. 线稿叠加 - 使用语义路由
    has_lineart = any(
        routing.region_configs.get(bucket, None) and 
        getattr(routing.region_configs.get(bucket), "lineart_strength", 0) > 0.01
        for bucket in unified_seg_out.semantic_masks.keys()
    )
    
    if has_lineart:
        fused = pipe.lineart.overlay_with_semantic_routing(
            image=fused,
            semantic_masks=unified_seg_out.semantic_masks,
            region_configs=routing.region_configs,
            params=ui_params
        )
    elif ui_params.get("edge_strength", 0) > 1e-3:
        edges = pipe.lineart.extract_from_stylized(fused, ui_params)
        fused = pipe.lineart.overlay(fused, edges, ui_params["edge_strength"], ui_params)
    
    # G2. 细节增强 - 使用语义路由
    has_detail = any(
        routing.region_configs.get(bucket, None) and 
        getattr(routing.region_configs.get(bucket), "detail_enhance", 0) > 0.01
        for bucket in unified_seg_out.semantic_masks.keys()
    )
    
    if has_detail:
        fused = pipe.lineart.enhance_detail_with_semantic_routing(
            image=fused,
            guide=original_image,
            semantic_masks=unified_seg_out.semantic_masks,
            region_configs=routing.region_configs
        )
    elif ui_params.get("detail_enhance_enabled", False):
        fused = pipe.lineart.enhance_detail(
            fused, original_image, ui_params.get("detail_strength", 0.5)
        )
    
    # Stage 3: 色调调整（最轻量）
    fused = apply_tone_adjustment(
        fused,
        ui_params.get("gamma", 1.0),
        ui_params.get("contrast", 1.0),
        ui_params.get("saturation", 1.0),
        ui_params.get("brightness", 0.0)
    )
    
    # 后处理
    out_u8 = pipe.preprocessor.postprocess(fused, ctx)
    return out_u8


def visualize_semantic_mask(
    state: ProcessingState,
    bucket: str,
    toggle: bool = True
) -> tuple[np.ndarray | None, str, ProcessingState]:
    """
    可视化指定语义区域的遮罩（支持叠加多个区域）
    
    Args:
        state: 用户会话状态
        bucket: 语义桶名称 或 "FACE" 或 "NONE"
        toggle: 是否切换该区域的显示状态
    
    Returns:
        (叠加遮罩后的图像, 覆盖率信息, 更新后的状态)
    """
    import cv2
    
    if state.original_image is None or state.seg_out is None:
        return None, "请先上传并处理图像", state
    
    new_state = state.copy()
    
    # 处理 NONE（清除所有遮罩）
    if bucket == "NONE":
        new_state.active_masks = set()
        return state.original_image.copy(), "显示原图", new_state
    
    # 切换该区域的激活状态
    if toggle:
        if bucket in new_state.active_masks:
            new_state.active_masks.discard(bucket)
        else:
            new_state.active_masks.add(bucket)
    
    # 如果没有激活的遮罩，返回原图
    if not new_state.active_masks:
        return state.original_image.copy(), "点击区域按钮查看遮罩", new_state
    
    # 获取原图
    original = state.original_image.copy()
    H, W = original.shape[:2]
    result = original.astype(np.float32)
    
    info_parts = []
    
    # 叠加所有激活的遮罩
    for active_bucket in new_state.active_masks:
        # 获取遮罩
        if active_bucket == "FACE":
            if state.face_mask is None:
                continue
            mask = state.face_mask
        else:
            if active_bucket not in state.seg_out.semantic_masks:
                continue
            mask = state.seg_out.semantic_masks[active_bucket]
        
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
    
    return result, info, new_state

