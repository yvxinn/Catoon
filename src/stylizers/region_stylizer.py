"""
RegionStylizer - 区域级风格化器

为每个语义区域生成独立的风格化图像，支持不同的 K 值和风格参数。

优化策略：
1. 懒加载：仅在需要时生成区域风格
2. 缓存：相同参数的区域风格被缓存
3. 并行：可选的多线程处理（CPU 密集型操作）
4. 增量：只重新生成参数变化的区域
"""

import numpy as np
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import DictConfig

from ..context import StyleCandidate, RegionConfig, SegmentationOutput
from .traditional import TraditionalStylizer
from .base import BaseStylizer


class RegionStylizer:
    """区域级风格化器"""
    
    def __init__(self, cfg: DictConfig, stylizers: dict[str, BaseStylizer]):
        """
        初始化区域风格化器
        
        Args:
            cfg: 配置对象
            stylizers: {style_id: stylizer} 风格化器字典
        """
        self.cfg = cfg
        self.stylizers = stylizers
        
        # 区域风格缓存: {cache_key: StyleCandidate}
        self._region_cache: dict[str, StyleCandidate] = {}
        
        # 当前图像哈希（用于缓存失效）
        self._current_image_hash: str = ""
        
        # 并行配置
        self.max_workers = cfg.get("region_stylizer", {}).get("max_workers", 4)
        self.enable_parallel = cfg.get("region_stylizer", {}).get("enable_parallel", True)
    
    def generate_region_styles(
        self,
        image_f32: np.ndarray,
        image_hash: str,
        seg_out: SegmentationOutput,
        region_configs: dict[str, RegionConfig],
        global_candidates: dict[str, StyleCandidate] | None = None
    ) -> dict[str, StyleCandidate]:
        """
        为每个区域生成风格化图像
        
        Args:
            image_f32: 原图 float32 (H,W,3) [0,1]
            image_hash: 图像哈希，用于缓存
            seg_out: 分割输出
            region_configs: {region_name: RegionConfig} 区域配置
            global_candidates: 全局候选（用于非 Traditional 风格）
        
        Returns:
            {region_name: StyleCandidate} 区域风格候选
        """
        # 检查图像是否变化，清理缓存
        if image_hash != self._current_image_hash:
            self._region_cache.clear()
            self._current_image_hash = image_hash
        
        region_styles: dict[str, StyleCandidate] = {}
        tasks_to_generate: list[tuple[str, RegionConfig]] = []
        
        # 检查缓存，收集需要生成的任务
        for region_name, config in region_configs.items():
            # 检查区域是否存在
            mask = seg_out.semantic_masks.get(region_name)
            if mask is None or mask.sum() < 100:  # 忽略过小的区域
                continue
            
            cache_key = self._make_cache_key(region_name, config)
            
            if cache_key in self._region_cache:
                # 使用缓存
                region_styles[region_name] = self._region_cache[cache_key]
            else:
                # 需要生成
                tasks_to_generate.append((region_name, config))
        
        if not tasks_to_generate:
            return region_styles
        
        # 生成区域风格
        if self.enable_parallel and len(tasks_to_generate) > 1:
            # 并行生成
            new_styles = self._generate_parallel(
                image_f32, tasks_to_generate, global_candidates
            )
        else:
            # 串行生成
            new_styles = self._generate_sequential(
                image_f32, tasks_to_generate, global_candidates
            )
        
        # 更新缓存和结果
        for region_name, candidate in new_styles.items():
            config = region_configs[region_name]
            cache_key = self._make_cache_key(region_name, config)
            self._region_cache[cache_key] = candidate
            region_styles[region_name] = candidate
        
        return region_styles
    
    def _generate_sequential(
        self,
        image_f32: np.ndarray,
        tasks: list[tuple[str, RegionConfig]],
        global_candidates: dict[str, StyleCandidate] | None
    ) -> dict[str, StyleCandidate]:
        """串行生成区域风格"""
        results = {}
        for region_name, config in tasks:
            candidate = self._generate_single(
                image_f32, region_name, config, global_candidates
            )
            if candidate is not None:
                results[region_name] = candidate
        return results
    
    def _generate_parallel(
        self,
        image_f32: np.ndarray,
        tasks: list[tuple[str, RegionConfig]],
        global_candidates: dict[str, StyleCandidate] | None
    ) -> dict[str, StyleCandidate]:
        """并行生成区域风格"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_single,
                    image_f32, region_name, config, global_candidates
                ): region_name
                for region_name, config in tasks
            }
            
            for future in as_completed(futures):
                region_name = futures[future]
                try:
                    candidate = future.result()
                    if candidate is not None:
                        results[region_name] = candidate
                except Exception as e:
                    print(f"[RegionStylizer] Error generating {region_name}: {e}")
        
        return results
    
    def _generate_single(
        self,
        image_f32: np.ndarray,
        region_name: str,
        config: RegionConfig,
        global_candidates: dict[str, StyleCandidate] | None
    ) -> StyleCandidate | None:
        """
        生成单个区域的风格化图像
        
        Args:
            image_f32: 原图
            region_name: 区域名称
            config: 区域配置
            global_candidates: 全局候选
        
        Returns:
            StyleCandidate 或 None
        """
        style_id = config.style_id
        
        if style_id == "Traditional":
            # Traditional 风格：使用区域级 K 值
            stylizer = self.stylizers.get("Traditional")
            if stylizer is None:
                return None
            
            # 使用区域配置的 K 值
            candidate = stylizer.stylize(
                image_f32,
                K=config.toon_K,
                smooth_strength=config.smooth_strength
            )
            
            # 更新 style_id 以标识区域
            return StyleCandidate(
                style_id=f"{style_id}_K{config.toon_K}",
                image=candidate.image,
                color_stats=candidate.color_stats,
                model_type=candidate.model_type,
                model_name=candidate.model_name
            )
        
        else:
            # GAN 风格：直接使用全局候选
            if global_candidates and style_id in global_candidates:
                return global_candidates[style_id]
            return None
    
    def _make_cache_key(self, region_name: str, config: RegionConfig) -> str:
        """生成缓存键"""
        # 对于 Traditional，K 值影响输出
        if config.style_id == "Traditional":
            return f"{region_name}_{config.style_id}_K{config.toon_K}_S{config.smooth_strength:.2f}"
        else:
            # GAN 风格不受区域参数影响
            return f"{region_name}_{config.style_id}"
    
    def clear_cache(self):
        """清空缓存"""
        self._region_cache.clear()
        self._current_image_hash = ""
    
    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self._region_cache),
            "image_hash": self._current_image_hash,
            "cached_regions": list(self._region_cache.keys())
        }


def create_region_stylizer(
    cfg: DictConfig,
    stylizers: dict[str, BaseStylizer]
) -> RegionStylizer:
    """
    便捷函数：创建区域风格化器
    
    Args:
        cfg: 配置对象
        stylizers: 风格化器字典
    
    Returns:
        RegionStylizer 实例
    """
    return RegionStylizer(cfg, stylizers)

