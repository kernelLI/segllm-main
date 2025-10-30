"""
LIDC-IDRI纵向推理分割变化计算模块
基于现有metrics_longitudinal.py进行扩展，支持纵向推理分割任务
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage import measure, morphology
import SimpleITK as sitk

# 导入现有的变化计算模块
from llava.metrics_longitudinal import ChangeCalculator, ChangeMetrics

logger = logging.getLogger(__name__)

@dataclass
class LongitudinalChangeMetrics:
    """纵向变化指标数据类"""
    # 体积变化
    volume_change_percent: float
    volume_change_mm3: float
    volume_ratio: float
    
    # 密度变化
    density_change_hu: float
    density_change_percent: float
    mean_hu_t0: float
    mean_hu_t1: float
    
    # 形态变化
    diameter_change_mm: float
    diameter_change_percent: float
    sphericity_change: float
    margin_blur_change: float
    
    # 空间变化
    centroid_shift_mm: float
    spatial_overlap_iou: float
    
    # 综合评分
    overall_progression_score: float  # 0-1, 0=稳定, 1=显著进展
    change_severity: str  # "stable", "mild", "moderate", "severe"
    
    # 医学意义
    clinical_significance: str  # "no_change", "progression", "regression"
    follow_up_recommendation: str

class LongitudinalChangeCalculator(ChangeCalculator):
    """扩展的纵向变化计算器"""
    
    def __init__(
        self,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        volume_threshold_percent: float = 25.0,
        density_threshold_hu: float = 50.0,
        diameter_threshold_mm: float = 2.0,
        clinical_progression_threshold: float = 0.7
    ):
        super().__init__(voxel_spacing=voxel_spacing)
        self.volume_threshold_percent = volume_threshold_percent
        self.density_threshold_hu = density_threshold_hu
        self.diameter_threshold_mm = diameter_threshold_mm
        self.clinical_progression_threshold = clinical_progression_threshold
        
        # 通用纵向变化阈值（对齐任务.md）
        self.clinical_thresholds = {
            "volume_progression": 20.0,    # 体积增加≥20%认为进展
            "volume_regression": -15.0,    # 体积减少≥15%认为退缩
            "density_progression": 30.0,   # 密度增加≥30HU认为进展
            "density_regression": -25.0,   # 密度减少≥25HU认为退缩
            "diameter_progression": 2.0,   # 直径增加≥2mm认为进展
            "centroid_shift_significant": 5.0  # 质心移动≥5mm认为显著
        }
    
    def compute_longitudinal_changes(
        self,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        nodule_id: Optional[str] = None,
        compute_textural: bool = True,
        compute_clinical: bool = True
    ) -> LongitudinalChangeMetrics:
        """计算完整的纵向变化指标"""
        
        # 基础变化计算
        volume_changes = self.calculate_volume_change(mask_t0, mask_t1)
        density_changes = self.calculate_density_change(ct_t0, ct_t1, mask_t0, mask_t1)
        diameter_changes = self.calculate_diameter_change(mask_t0, mask_t1)
        
        # 扩展的形态学变化
        morphological_changes = self._compute_advanced_morphological_changes(
            mask_t0, mask_t1, ct_t0, ct_t1)
        
        # 空间变化
        spatial_changes = self._compute_spatial_changes(mask_t0, mask_t1)
        
        # 纹理变化（可选）
        textural_changes = {}
        if compute_textural:
            textural_changes = self._compute_textural_changes(ct_t0, ct_t1, mask_t0, mask_t1)
        
        # 临床意义评估
        clinical_assessment = {}
        if compute_clinical:
            clinical_assessment = self._assess_clinical_significance(
                volume_changes, density_changes, diameter_changes, spatial_changes)
        
        # 综合评分
        overall_score = self._compute_overall_progression_score(
            volume_changes, density_changes, morphological_changes, spatial_changes)
        
        # 构建完整的变化指标
        change_metrics = LongitudinalChangeMetrics(
            # 体积变化
            volume_change_percent=volume_changes["volume_change_percent"],
            volume_change_mm3=volume_changes["volume_change_mm3"],
            volume_ratio=volume_changes["volume_change_ratio"],
            
            # 密度变化
            density_change_hu=density_changes["density_change_hu"],
            density_change_percent=density_changes["density_change_percent"],
            mean_hu_t0=density_changes["mean_hu_t0"],
            mean_hu_t1=density_changes["mean_hu_t1"],
            
            # 形态变化
            diameter_change_mm=diameter_changes["diameter_change_mm"],
            diameter_change_percent=diameter_changes["diameter_change_percent"],
            sphericity_change=morphological_changes.get("sphericity_change", 0.0),
            margin_blur_change=morphological_changes.get("margin_blur_change", 0.0),
            
            # 空间变化
            centroid_shift_mm=spatial_changes["centroid_shift_mm"],
            spatial_overlap_iou=spatial_changes["spatial_overlap_iou"],
            
            # 综合评分
            overall_progression_score=overall_score,
            change_severity=self._classify_change_severity(overall_score),
            
            # 医学意义
            clinical_significance=clinical_assessment.get("clinical_significance", "unknown"),
            follow_up_recommendation=clinical_assessment.get("follow_up_recommendation", "routine")
        )
        
        return change_metrics
    
    def _compute_advanced_morphological_changes(
        self, mask_t0: np.ndarray, mask_t1: np.ndarray,
        ct_t0: np.ndarray, ct_t1: np.ndarray
    ) -> Dict[str, float]:
        """计算高级形态学变化"""
        
        # 球形度变化
        sphericity_t0 = self._calculate_sphericity(mask_t0)
        sphericity_t1 = self._calculate_sphericity(mask_t1)
        sphericity_change = sphericity_t1 - sphericity_t0
        
        # 边界模糊度变化
        margin_blur_t0 = self._calculate_margin_blur(mask_t0, ct_t0)
        margin_blur_t1 = self._calculate_margin_blur(mask_t1, ct_t1)
        margin_blur_change = margin_blur_t1 - margin_blur_t0
        
        # 表面积变化
        surface_t0 = self._calculate_surface_area(mask_t0)
        surface_t1 = self._calculate_surface_area(mask_t1)
        surface_change = ((surface_t1 - surface_t0) / max(surface_t0, 1e-6)) * 100
        
        # 紧凑度变化
        compactness_t0 = self._calculate_compactness(mask_t0)
        compactness_t1 = self._calculate_compactness(mask_t1)
        compactness_change = compactness_t1 - compactness_t0
        
        return {
            "sphericity_change": sphericity_change,
            "margin_blur_change": margin_blur_change,
            "surface_change_percent": surface_change,
            "compactness_change": compactness_change,
            "sphericity_t0": sphericity_t0,
            "sphericity_t1": sphericity_t1
        }
    
    def _calculate_sphericity(self, mask: np.ndarray) -> float:
        """计算球形度"""
        volume = np.sum(mask > 0) * np.prod(self.voxel_spacing)
        surface_area = self._calculate_surface_area(mask)
        
        if surface_area <= 0:
            return 0.0
        
        # 球形度 = (36πV²)^(1/3) / A
        sphericity = ((36 * np.pi * volume**2) ** (1/3)) / surface_area
        return min(sphericity, 1.0)  # 限制在0-1之间
    
    def _calculate_margin_blur(self, mask: np.ndarray, ct_image: np.ndarray) -> float:
        """计算边界模糊度"""
        # 获取边界
        boundary = morphology.binary_dilation(mask, morphology.ball(1)) - mask
        
        if np.sum(boundary) == 0:
            return np.nan
        
        # 计算边界区域的梯度
        boundary_ct = ct_image[boundary > 0]
        if len(boundary_ct) == 0:
            return np.nan
        
        # 如果 HU 全 0（伪 HU）或全常数，返回 NaN 避免误导
        if np.allclose(boundary_ct, 0) or np.std(boundary_ct) < 1e-6:
            return np.nan
        
        # 边界模糊度 = 边界区域HU值的标准差
        margin_blur = np.std(boundary_ct)
        return margin_blur
    
    def _calculate_surface_area(self, mask: np.ndarray) -> float:
        """计算表面积"""
        # 使用Marching Cubes算法计算表面积
        try:
            from skimage import measure
            verts, faces = measure.marching_cubes(mask, level=0.5)
            surface_area = measure.mesh_surface_area(verts, faces)
            return surface_area * (self.voxel_spacing[0] ** 2)  # 转换为mm²
        except:
            # 简化方法：计算边界体素数量
            boundary = morphology.binary_dilation(mask, morphology.ball(1)) - mask
            return np.sum(boundary) * (self.voxel_spacing[0] ** 2)
    
    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """计算紧凑度"""
        labeled = measure.label(mask > 0)
        if labeled.max() == 0:
            return 0.0
        
        regions = measure.regionprops(labeled)
        if not regions:
            return 0.0
        
        largest_region = max(regions, key=lambda x: x.area)
        
        # 紧凑度 = 面积 / (周长²)
        perimeter = largest_region.perimeter
        if perimeter <= 0:
            return 0.0
        
        compactness = largest_region.area / (perimeter ** 2)
        return compactness
    
    def _compute_spatial_changes(self, mask_t0: np.ndarray, mask_t1: np.ndarray) -> Dict[str, float]:
        """计算空间变化"""
        
        # 计算质心
        centroid_t0 = self._calculate_centroid(mask_t0)
        centroid_t1 = self._calculate_centroid(mask_t1)
        
        # 质心移动距离
        centroid_shift = np.sqrt(np.sum((np.array(centroid_t0) - np.array(centroid_t1))**2))
        centroid_shift_mm = centroid_shift * self.voxel_spacing[0]
        
        # 空间重叠度（IoU）
        intersection = np.sum((mask_t0 > 0) & (mask_t1 > 0))
        union = np.sum((mask_t0 > 0) | (mask_t1 > 0))
        spatial_iou = intersection / max(union, 1e-6)
        
        # 形状变化（Hausdorff距离）
        try:
            from scipy.spatial.distance import directed_hausdorff
            hausdorff_distance = max(
                directed_hausdorff(np.argwhere(mask_t0 > 0), np.argwhere(mask_t1 > 0))[0],
                directed_hausdorff(np.argwhere(mask_t1 > 0), np.argwhere(mask_t0 > 0))[0]
            )
            hausdorff_mm = hausdorff_distance * self.voxel_spacing[0]
        except:
            hausdorff_mm = 0.0
        
        return {
            "centroid_shift_mm": centroid_shift_mm,
            "spatial_overlap_iou": spatial_iou,
            "hausdorff_distance_mm": hausdorff_mm
        }
    
    def _calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """计算质心"""
        if np.sum(mask) == 0:
            return (0.0, 0.0, 0.0)
        
        indices = np.argwhere(mask > 0)
        centroid = np.mean(indices, axis=0)
        return tuple(centroid)
    
    def _compute_textural_changes(
        self, ct_t0: np.ndarray, ct_t1: np.ndarray,
        mask_t0: np.ndarray, mask_t1: np.ndarray
    ) -> Dict[str, float]:
        """计算纹理变化"""
        
        # 计算灰度共生矩阵特征
        def compute_glcm_features(image, mask):
            from skimage.feature import graycomatrix, graycoprops
            
            # 提取ROI
            roi = image[mask > 0]
            if len(roi) == 0:
                return {"contrast": 0.0, "homogeneity": 0.0, "energy": 0.0}
            
            # 量化到8位
            roi_quantized = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-6) * 7).astype(np.uint8)
            
            # 计算GLCM
            glcm = graycomatrix(roi_quantized, distances=[1], angles=[0], levels=8)
            
            return {
                "contrast": graycoprops(glcm, 'contrast')[0, 0],
                "homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
                "energy": graycoprops(glcm, 'energy')[0, 0]
            }
        
        # 计算纹理特征
        features_t0 = compute_glcm_features(ct_t0, mask_t0)
        features_t1 = compute_glcm_features(ct_t1, mask_t1)
        
        # 纹理变化
        textural_changes = {}
        for feature_name in features_t0.keys():
            change = features_t1[feature_name] - features_t0[feature_name]
            textural_changes[f"{feature_name}_change"] = change
            textural_changes[f"{feature_name}_ratio"] = features_t1[feature_name] / max(features_t0[feature_name], 1e-6)
        
        return textural_changes
    
    def _assess_clinical_significance(
        self, volume_changes: Dict[str, float], density_changes: Dict[str, float],
        diameter_changes: Dict[str, float], spatial_changes: Dict[str, float]
    ) -> Dict[str, str]:
        """评估临床意义"""
        
        volume_change = volume_changes["volume_change_percent"]
        density_change = density_changes["density_change_hu"]
        diameter_change = diameter_changes["diameter_change_mm"]
        centroid_shift = spatial_changes["centroid_shift_mm"]
        
        # 判断临床意义
        clinical_significance = "no_change"
        if volume_change >= self.clinical_thresholds["volume_progression"]:
            clinical_significance = "progression"
        elif volume_change <= self.clinical_thresholds["volume_regression"]:
            clinical_significance = "regression"
        elif abs(density_change) >= self.clinical_thresholds["density_progression"]:
            clinical_significance = "progression"
        elif diameter_change >= self.clinical_thresholds["diameter_progression"]:
            clinical_significance = "progression"
        
        # 随访建议
        follow_up_recommendation = "routine"
        if clinical_significance == "progression":
            follow_up_recommendation = "immediate_followup"
        elif clinical_significance == "regression":
            follow_up_recommendation = "routine_followup"
        elif abs(volume_change) > 10 or abs(density_change) > 20:
            follow_up_recommendation = "short_term_followup"
        
        return {
            "clinical_significance": clinical_significance,
            "follow_up_recommendation": follow_up_recommendation
        }
    
    def _compute_overall_progression_score(
        self, volume_changes: Dict[str, float], density_changes: Dict[str, float],
        morphological_changes: Dict[str, float], spatial_changes: Dict[str, float]
    ) -> float:
        """计算综合进展评分"""
        
        volume_change = abs(volume_changes["volume_change_percent"]) / 100.0
        density_change = abs(density_changes["density_change_hu"]) / 1000.0
        morphological_change = abs(morphological_changes.get("sphericity_change", 0.0))
        spatial_change = spatial_changes["centroid_shift_mm"] / 50.0
        
        # 加权综合评分
        weights = {
            "volume": 0.4,
            "density": 0.25,
            "morphological": 0.2,
            "spatial": 0.15
        }
        
        normalized_changes = {
            "volume": min(volume_change, 1.0),
            "density": min(density_change, 1.0),
            "morphological": min(morphological_change, 1.0),
            "spatial": min(spatial_change, 1.0)
        }
        
        overall_score = sum(
            weights[key] * normalized_changes[key] 
            for key in weights.keys()
        )
        
        return min(overall_score, 1.0)
    
    def _classify_change_severity(self, overall_score: float) -> str:
        """分类变化严重程度"""
        if overall_score < 0.2:
            return "stable"
        elif overall_score < 0.4:
            return "mild"
        elif overall_score < 0.7:
            return "moderate"
        else:
            return "severe"
    
    def compute_changes_between_timepoints(
        self, 
        timepoint_data: List[Dict[str, Any]], 
        compute_textural: bool = True,
        compute_clinical: bool = True
    ) -> List[LongitudinalChangeMetrics]:
        """批量计算多个时相之间的变化"""
        
        if len(timepoint_data) < 2:
            logger.warning("Need at least 2 timepoints to compute changes")
            return []
        
        change_metrics_list = []
        
        for i in range(len(timepoint_data) - 1):
            t0_data = timepoint_data[i]
            t1_data = timepoint_data[i + 1]
            
            # 提取数据
            mask_t0 = t0_data["mask"]
            mask_t1 = t1_data["mask"]
            ct_t0 = t0_data["ct_image"]
            ct_t1 = t1_data["ct_image"]
            nodule_id = t0_data.get("nodule_id", f"nodule_{i}")
            
            # 计算变化
            changes = self.compute_longitudinal_changes(
                mask_t0, mask_t1, ct_t0, ct_t1,
                nodule_id=nodule_id,
                compute_textural=compute_textural,
                compute_clinical=compute_clinical
            )
            
            change_metrics_list.append(changes)
        
        return change_metrics_list

# 便捷函数
def compute_longitudinal_changes(
    mask_t0: np.ndarray,
    mask_t1: np.ndarray,
    ct_t0: np.ndarray,
    ct_t1: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    **kwargs
) -> LongitudinalChangeMetrics:
    """便捷函数：计算纵向变化"""
    calculator = LongitudinalChangeCalculator(voxel_spacing=voxel_spacing, **kwargs)
    return calculator.compute_longitudinal_changes(mask_t0, mask_t1, ct_t0, ct_t1, **kwargs)

def batch_compute_changes(timepoint_data: List[Dict[str, Any]], **kwargs) -> List[LongitudinalChangeMetrics]:
    """便捷函数：批量计算变化"""
    if not timepoint_data:
        return []
    
    # 从第一个样本获取体素间距
    first_sample = timepoint_data[0]
    voxel_spacing = first_sample.get("voxel_spacing", (1.0, 1.0, 1.0))
    
    calculator = LongitudinalChangeCalculator(voxel_spacing=voxel_spacing, **kwargs)
    return calculator.compute_changes_between_timepoints(timepoint_data, **kwargs)

if __name__ == "__main__":
    # 示例用法
    # 创建示例数据
    mask_t0 = np.random.rand(64, 64, 32) > 0.7
    mask_t1 = np.random.rand(64, 64, 32) > 0.6
    ct_t0 = np.random.randint(-1000, 400, (64, 64, 32))
    ct_t1 = np.random.randint(-1000, 400, (64, 64, 32))
    
    # 计算变化
    changes = compute_longitudinal_changes(mask_t0, mask_t1, ct_t0, ct_t1)
    
    print(f"Volume change: {changes.volume_change_percent:.1f}%")
    print(f"Density change: {changes.density_change_hu:.1f} HU")
    print(f"Overall progression score: {changes.overall_progression_score:.3f}")
    print(f"Clinical significance: {changes.clinical_significance}")
    print(f"Follow-up recommendation: {changes.follow_up_recommendation}")