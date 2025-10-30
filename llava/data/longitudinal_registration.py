"""
LIDC-IDRI多时相CT扫描配对和结节配准模块
基于现有nodule_registration.py进行扩展，支持纵向推理分割任务
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage import measure, morphology
import SimpleITK as sitk

# 导入现有的配准模块
from llava.data.nodule_registration import NoduleRegistration, NoduleInfo

logger = logging.getLogger(__name__)

@dataclass
class TemporalPair:
    """时相配对数据类"""
    patient_id: str
    baseline_study: Dict[str, Any]
    followup_study: Dict[str, Any]
    baseline_date: str
    followup_date: str
    time_interval_days: int
    registration_transform: Optional[Any] = None
    registration_quality: float = 0.0

@dataclass
class MatchedNodulePair:
    """匹配的结节对"""
    baseline_nodule: NoduleInfo
    followup_nodule: NoduleInfo
    match_score: float
    iou_score: float
    spatial_distance: float
    morphological_similarity: float
    temporal_changes: Dict[str, float]

class TemporalCTPairing:
    """多时相CT扫描配对器"""
    
    def __init__(
        self,
        max_time_interval_days: int = 730,  # 2年
        min_time_interval_days: int = 30,   # 1个月
        registration_method: str = "bspline",
        quality_threshold: float = 0.7
    ):
        self.max_time_interval_days = max_time_interval_days
        self.min_time_interval_days = min_time_interval_days
        self.registration_method = registration_method
        self.quality_threshold = quality_threshold
        
    def find_temporal_pairs(self, patient_studies: Dict[str, List[Dict[str, Any]]]) -> List[TemporalPair]:
        """查找患者的时相配对"""
        temporal_pairs = []
        
        for patient_id, studies in patient_studies.items():
            if len(studies) < 2:
                continue
                
            # 按日期排序
            studies_sorted = sorted(studies, key=lambda x: x.get('study_date', ''))
            
            # 生成所有可能的配对
            for i in range(len(studies_sorted)):
                for j in range(i + 1, len(studies_sorted)):
                    baseline = studies_sorted[i]
                    followup = studies_sorted[j]
                    
                    # 检查时间间隔
                    time_diff = self._calculate_time_diff(
                        baseline.get('study_date', ''),
                        followup.get('study_date', '')
                    )
                    
                    if (self.min_time_interval_days <= time_diff <= self.max_time_interval_days):
                        pair = TemporalPair(
                            patient_id=patient_id,
                            baseline_study=baseline,
                            followup_study=followup,
                            baseline_date=baseline.get('study_date', ''),
                            followup_date=followup.get('study_date', ''),
                            time_interval_days=time_diff
                        )
                        temporal_pairs.append(pair)
        
        return temporal_pairs
    
    def _calculate_time_diff(self, date1: str, date2: str) -> int:
        """计算两个日期之间的天数差"""
        try:
            # 支持多种日期格式
            for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    d1 = datetime.strptime(date1, fmt)
                    d2 = datetime.strptime(date2, fmt)
                    return abs((d2 - d1).days)
                except ValueError:
                    continue
            return 0
        except:
            return 0

class LongitudinalNoduleRegistration(NoduleRegistration):
    """扩展的结节配准类，支持纵向推理分割"""
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        distance_threshold: float = 30.0,
        morphological_weight: float = 0.3,
        temporal_weight: float = 0.2,
        **kwargs
    ):
        super().__init__(iou_threshold=iou_threshold, 
                        distance_threshold=distance_threshold,
                        morphological_weight=morphological_weight)
        self.temporal_weight = temporal_weight
        
    def register_temporal_nodules(
        self, 
        baseline_nodules: List[NoduleInfo], 
        followup_nodules: List[NoduleInfo],
        registration_transform: Optional[Any] = None
    ) -> List[MatchedNodulePair]:
        """配准时相结节"""
        # 应用空间变换（如果有）
        if registration_transform is not None:
            baseline_nodules = self._apply_transform_to_nodules(
                baseline_nodules, registration_transform)
        
        # 计算匹配矩阵
        cost_matrix = self._compute_temporal_matching_matrix(
            baseline_nodules, followup_nodules)
        
        # 使用匈牙利算法进行最优匹配
        baseline_indices, followup_indices = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        for bi, fi in zip(baseline_indices, followup_indices):
            if cost_matrix[bi, fi] < 0.5:  # 匹配质量阈值
                baseline_nodule = baseline_nodules[bi]
                followup_nodule = followup_nodules[fi]
                
                # 计算详细的匹配分数
                match_score = 1.0 - cost_matrix[bi, fi]
                iou_score = self._compute_iou(baseline_nodule.mask, followup_nodule.mask)
                spatial_distance = self._compute_spatial_distance(baseline_nodule, followup_nodule)
                morphological_similarity = self._compute_morphological_similarity(baseline_nodule, followup_nodule)
                
                # 计算时相变化
                temporal_changes = self._compute_temporal_changes(baseline_nodule, followup_nodule)
                
                pair = MatchedNodulePair(
                    baseline_nodule=baseline_nodule,
                    followup_nodule=followup_nodule,
                    match_score=match_score,
                    iou_score=iou_score,
                    spatial_distance=spatial_distance,
                    morphological_similarity=morphological_similarity,
                    temporal_changes=temporal_changes
                )
                matched_pairs.append(pair)
        
        return matched_pairs
    
    def _compute_temporal_matching_matrix(self, baseline_nodules: List[NoduleInfo], 
                                        followup_nodules: List[NoduleInfo]) -> np.ndarray:
        """计算时相匹配代价矩阵"""
        n_baseline = len(baseline_nodules)
        n_followup = len(followup_nodules)
        cost_matrix = np.ones((n_baseline, n_followup))
        
        for i, baseline in enumerate(baseline_nodules):
            for j, followup in enumerate(followup_nodules):
                # 空间距离代价
                spatial_cost = self._compute_spatial_distance_cost(baseline, followup)
                
                # IOU代价
                iou_cost = 1.0 - self._compute_iou(baseline.mask, followup.mask)
                
                # 形态学代价
                morph_cost = 1.0 - self._compute_morphological_similarity(baseline, followup)
                
                # 时相一致性代价（体积变化合理性）
                temporal_cost = self._compute_temporal_consistency_cost(baseline, followup)
                
                # 综合代价
                cost_matrix[i, j] = (
                    spatial_cost * (1 - self.iou_weight - self.morphological_weight - self.temporal_weight) +
                    iou_cost * self.iou_weight +
                    morph_cost * self.morphological_weight +
                    temporal_cost * self.temporal_weight
                )
        
        return cost_matrix
    
    def _compute_temporal_consistency_cost(self, baseline: NoduleInfo, followup: NoduleInfo) -> float:
        """计算时相一致代价（体积变化合理性）"""
        if baseline.volume <= 0:
            return 1.0
        
        volume_ratio = followup.volume / baseline.volume
        
        # 体积变化应该在合理范围内（0.1到10倍）
        if volume_ratio < 0.1 or volume_ratio > 10.0:
            return 1.0
        
        # 体积变化越小，代价越小
        return min(abs(volume_ratio - 1.0), 1.0)
    
    def _compute_temporal_changes(self, baseline: NoduleInfo, followup: NoduleInfo) -> Dict[str, float]:
        """计算时相变化指标"""
        # 体积变化
        volume_change = ((followup.volume - baseline.volume) / max(baseline.volume, 1e-6)) * 100
        
        # 密度变化
        density_change = followup.mean_hu - baseline.mean_hu
        
        # 直径变化
        diameter_change = followup.diameter - baseline.diameter
        
        # 球形度变化
        sphericity_change = followup.sphericity - baseline.sphericity
        
        return {
            "volume_change_percent": volume_change,
            "density_change_hu": density_change,
            "diameter_change_mm": diameter_change,
            "sphericity_change": sphericity_change,
            "volume_ratio": followup.volume / max(baseline.volume, 1e-6),
            "density_ratio": followup.mean_hu / max(abs(baseline.mean_hu), 1e-6)
        }
    
    def _apply_transform_to_nodules(self, nodules: List[NoduleInfo], 
                                  transform: Any) -> List[NoduleInfo]:
        """将变换应用到结节"""
        transformed_nodules = []
        
        for nodule in nodules:
            # 应用变换到掩码
            transformed_mask = self._apply_transform_to_mask(nodule.mask, transform)
            
            # 创建新的结节信息
            transformed_nodule = NoduleInfo(
                nodule_id=nodule.nodule_id,
                mask=transformed_mask,
                centroid=self._calculate_centroid(transformed_mask),
                volume=self._calculate_volume(transformed_mask),
                mean_hu=nodule.mean_hu,  # 密度值不变
                diameter=self._calculate_diameter(transformed_mask),
                sphericity=self._calculate_sphericity(transformed_mask)
            )
            transformed_nodules.append(transformed_nodule)
        
        return transformed_nodules
    
    def _apply_transform_to_mask(self, mask: np.ndarray, transform: Any) -> np.ndarray:
        """应用变换到掩码"""
        # 使用SimpleITK进行变换
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        if isinstance(transform, sitk.Transform):
            resampler = sitk.ResampleImageFilter()
            resampler.SetTransform(transform)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetReferenceImage(mask_sitk)
            transformed_mask_sitk = resampler.Execute(mask_sitk)
            transformed_mask = sitk.GetArrayFromImage(transformed_mask_sitk)
        else:
            transformed_mask = mask  # 无变换
        
        return transformed_mask > 0

class LongitudinalRegistrationPipeline:
    """纵向配准完整流程"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pairing = TemporalCTPairing(
            max_time_interval_days=config.get('max_time_interval_days', 730),
            min_time_interval_days=config.get('min_time_interval_days', 30),
            registration_method=config.get('registration_method', 'bspline'),
            quality_threshold=config.get('quality_threshold', 0.7)
        )
        self.registration = LongitudinalNoduleRegistration(
            iou_threshold=config.get('iou_threshold', 0.3),
            distance_threshold=config.get('distance_threshold', 30.0),
            morphological_weight=config.get('morphological_weight', 0.3),
            temporal_weight=config.get('temporal_weight', 0.2)
        )
    
    def process_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个患者的纵向数据"""
        # 1. 查找时相配对
        temporal_pairs = self.pairing.find_temporal_pairs({
            patient_data['patient_id']: patient_data['studies']
        })
        
        results = []
        for pair in temporal_pairs:
            # 2. 提取结节信息
            baseline_nodules = self._extract_nodules(pair.baseline_study)
            followup_nodules = self._extract_nodules(pair.followup_study)
            
            # 3. 进行图像配准（如果需要）
            registration_transform = None
            if self.config.get('enable_registration', True):
                registration_transform = self._register_images(
                    pair.baseline_study, pair.followup_study)
            
            # 4. 配准结节
            matched_pairs = self.registration.register_temporal_nodules(
                baseline_nodules, followup_nodules, registration_transform)
            
            # 5. 生成结果
            result = {
                'patient_id': pair.patient_id,
                'baseline_date': pair.baseline_date,
                'followup_date': pair.followup_date,
                'time_interval_days': pair.time_interval_days,
                'matched_nodules': self._format_matched_pairs(matched_pairs),
                'registration_quality': pair.registration_quality
            }
            results.append(result)
        
        return {
            'patient_id': patient_data['patient_id'],
            'temporal_pairs': results
        }
    
    def _extract_nodules(self, study_data: Dict[str, Any]) -> List[NoduleInfo]:
        """从研究数据中提取结节信息"""
        nodules = []
        for nodule_data in study_data.get('nodules', []):
            nodule = NoduleInfo(
                nodule_id=nodule_data['nodule_id'],
                mask=np.load(nodule_data['mask_path']),
                centroid=tuple(nodule_data['centroid']),
                volume=nodule_data['volume_mm3'],
                mean_hu=nodule_data['mean_hu'],
                diameter=nodule_data['diameter_mm'],
                sphericity=nodule_data.get('sphericity', 0.0)
            )
            nodules.append(nodule)
        return nodules
    
    def _register_images(self, baseline_study: Dict[str, Any], 
                        followup_study: Dict[str, Any]) -> Any:
        """进行图像配准"""
        # 这里简化处理，实际应用中需要实现完整的配准流程
        logger.info("Performing image registration...")
        return None  # 返回变换对象
    
    def _format_matched_pairs(self, matched_pairs: List[MatchedNodulePair]) -> List[Dict[str, Any]]:
        """格式化匹配的结节对"""
        formatted_pairs = []
        for pair in matched_pairs:
            formatted_pair = {
                'baseline_nodule_id': pair.baseline_nodule.nodule_id,
                'followup_nodule_id': pair.followup_nodule.nodule_id,
                'match_score': pair.match_score,
                'iou_score': pair.iou_score,
                'spatial_distance': pair.spatial_distance,
                'temporal_changes': pair.temporal_changes
            }
            formatted_pairs.append(formatted_pair)
        return formatted_pairs

def create_longitudinal_registration_pipeline(config: Dict[str, Any]) -> LongitudinalRegistrationPipeline:
    """创建纵向配准流程"""
    return LongitudinalRegistrationPipeline(config)

# 示例配置
DEFAULT_CONFIG = {
    'max_time_interval_days': 730,
    'min_time_interval_days': 30,
    'registration_method': 'bspline',
    'quality_threshold': 0.7,
    'iou_threshold': 0.3,
    'distance_threshold': 30.0,
    'morphological_weight': 0.3,
    'temporal_weight': 0.2,
    'enable_registration': True
}

if __name__ == "__main__":
    # 示例用法
    config = DEFAULT_CONFIG.copy()
    pipeline = create_longitudinal_registration_pipeline(config)
    
    # 示例患者数据
    patient_data = {
        'patient_id': 'LIDC-IDRI-0001',
        'studies': [
            {
                'study_date': '20200101',
                'nodules': [
                    {
                        'nodule_id': 'nodule_001',
                        'mask_path': '/path/to/mask1.npy',
                        'centroid': [100, 120, 50],
                        'volume_mm3': 150.0,
                        'mean_hu': -600.0,
                        'diameter_mm': 8.0,
                        'sphericity': 0.85
                    }
                ]
            },
            {
                'study_date': '20200601',
                'nodules': [
                    {
                        'nodule_id': 'nodule_001',
                        'mask_path': '/path/to/mask2.npy',
                        'centroid': [102, 118, 52],
                        'volume_mm3': 200.0,
                        'mean_hu': -550.0,
                        'diameter_mm': 9.0,
                        'sphericity': 0.87
                    }
                ]
            }
        ]
    }
    
    result = pipeline.process_patient(patient_data)
    print(f"Processed patient {result['patient_id']} with {len(result['temporal_pairs'])} temporal pairs")