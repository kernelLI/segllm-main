"""
LIDC-IDRI纵向推理分割数据生成器
基于现有SegLLM架构，实现LIDC-IDRI数据集的纵向推理分割任务
"""
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# 导入纵向配准模块
from llava.data.longitudinal_registration import (
    LongitudinalNoduleRegistration, 
    MatchedNodulePair, 
    TemporalCTPairing,
    create_longitudinal_registration_pipeline,
    DEFAULT_CONFIG
)

logger = logging.getLogger(__name__)

@dataclass
class NoduleInfo:
    """结节信息数据类"""
    nodule_id: str
    patient_id: str
    study_date: str
    series_uid: str
    mask_path: str
    image_path: str
    volume_mm3: float
    mean_hu: float
    diameter_mm: float
    centroid: Tuple[float, float, float]
    lesion_type: str  # solid, part_solid, ground_glass
    malignancy_score: Optional[float] = None

@dataclass
class LongitudinalPair:
    """纵向配对数据类"""
    patient_id: str
    baseline_date: str
    followup_date: str
    baseline_nodules: List[NoduleInfo]
    followup_nodules: List[NoduleInfo]
    matched_nodules: Dict[str, Tuple[NoduleInfo, NoduleInfo]]  # nodule_id -> (baseline, followup)
    changes: Dict[str, Dict[str, float]]  # nodule_id -> change_metrics
    registration_quality: float = 0.0  # 配准质量分数（0-1）

class LIDCLongitudinalDataGenerator:
    """LIDC-IDRI纵向数据生成器"""
    
    def __init__(
        self,
        data_root: str,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        volume_threshold: float = 25.0,  # 体积变化阈值(%)
        density_threshold: float = 50.0,  # 密度变化阈值(HU)
        min_nodule_volume: float = 30.0,  # 最小结节体积(mm³)
        max_followup_days: int = 730,  # 最大随访天数(2年)
        enable_registration: bool = True,  # 启用配准
        registration_config: Optional[Dict[str, Any]] = None,  # 配准配置
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.voxel_spacing = voxel_spacing
        self.volume_threshold = volume_threshold
        self.density_threshold = density_threshold
        self.min_nodule_volume = min_nodule_volume
        self.max_followup_days = max_followup_days
        self.enable_registration = enable_registration
        self.rng = random.Random(seed)
        
        # 初始化配准器
        if self.enable_registration:
            reg_config = registration_config or DEFAULT_CONFIG.copy()
            self.registration_pipeline = create_longitudinal_registration_pipeline(reg_config)
            self.nodule_registration = LongitudinalNoduleRegistration(
                iou_threshold=reg_config.get('iou_threshold', 0.3),
                distance_threshold=reg_config.get('distance_threshold', 30.0),
                morphological_weight=reg_config.get('morphological_weight', 0.3),
                temporal_weight=reg_config.get('temporal_weight', 0.2)
            )
        else:
            self.registration_pipeline = None
            self.nodule_registration = None
        
        # 任务类型定义
        self.task_types = {
            "volume_threshold": "体积阈值推理",
            "new_lesion": "新发/消退病灶",
            "density_change": "密度/形态变化",
            "combined_attributes": "多属性组合"
        }
        
        # 指令模板（中英文）
        self.instruction_templates = {
            "volume_threshold": {
                "zh": "[IMAGE256:{image_t0}|{image_t1}] 分割所有较上次体积增加超过{threshold}%的结节",
                "en": "[IMAGE256:{image_t0}|{image_t1}] Segment all nodules with volume increased by more than {threshold}% compared to previous scan"
            },
            "new_lesion": {
                "zh": "[IMAGE256:{image_t0}|{image_t1}] 标出新出现的{lesion_type}病灶",
                "en": "[IMAGE256:{image_t0}|{image_t1}] Identify newly appeared {lesion_type} lesions"
            },
            "density_change": {
                "zh": "[IMAGE256:{image_t0}|{image_t1}] 分割密度变化超过{threshold}HU的结节",
                "en": "[IMAGE256:{image_t0}|{image_t1}] Segment nodules with density change exceeding {threshold} HU"
            },
            "combined_attributes": {
                "zh": "[IMAGE256:{image_t0}|{image_t1}] 分割体积增加≥{vol_threshold}%且密度变化≥{den_threshold}HU的结节",
                "en": "[IMAGE256:{image_t0}|{image_t1}] Segment nodules with volume increase ≥{vol_threshold}% and density change ≥{den_threshold} HU"
            }
        }
    
    def load_lidc_data(self, metadata_file: str) -> Dict[str, List[NoduleInfo]]:
        """加载LIDC元数据"""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        patient_nodules = {}
        
        for item in metadata:
            patient_id = item.get('patient_id', '')
            study_date = item.get('study_date', '')
            
            nodule_info = NoduleInfo(
                nodule_id=item.get('nodule_id', ''),
                patient_id=patient_id,
                study_date=study_date,
                series_uid=item.get('series_uid', ''),
                mask_path=item.get('mask_path', ''),
                image_path=item.get('image_path', ''),
                volume_mm3=item.get('volume_mm3', 0.0),
                mean_hu=item.get('mean_hu', 0.0),
                diameter_mm=item.get('diameter_mm', 0.0),
                centroid=tuple(item.get('centroid', [0, 0, 0])),
                lesion_type=item.get('lesion_type', 'solid'),
                malignancy_score=item.get('malignancy_score')
            )
            
            if patient_id not in patient_nodules:
                patient_nodules[patient_id] = []
            patient_nodules[patient_id].append(nodule_info)
        
        return patient_nodules
    
    def find_longitudinal_pairs(self, patient_nodules: Dict[str, List[NoduleInfo]]) -> List[LongitudinalPair]:
        """查找纵向配对"""
        pairs = []
        
        for patient_id, nodules in patient_nodules.items():
            if len(nodules) < 2:
                continue
            
            # 按日期排序
            nodules.sort(key=lambda x: x.study_date)
            
            # 生成所有可能的配对
            for i in range(len(nodules) - 1):
                for j in range(i + 1, len(nodules)):
                    baseline = nodules[i]
                    followup = nodules[j]
                    
                    # 检查时间间隔
                    date_diff = self._calculate_date_diff(baseline.study_date, followup.study_date)
                    if date_diff > self.max_followup_days:
                        continue
                    
                    # 创建配对
                    pair = self._create_longitudinal_pair(patient_id, baseline, followup)
                    if pair:
                        pairs.append(pair)
        
        return pairs
    
    def _calculate_date_diff(self, date1: str, date2: str) -> int:
        """计算日期差（天数）"""
        try:
            d1 = datetime.strptime(date1, "%Y%m%d")
            d2 = datetime.strptime(date2, "%Y%m%d")
            return abs((d2 - d1).days)
        except:
            return 0
    
    def _create_longitudinal_pair(self, patient_id: str, baseline: NoduleInfo, followup: NoduleInfo) -> Optional[LongitudinalPair]:
        """创建纵向配对（集成配准功能）"""
        baseline_nodules = [baseline]
        followup_nodules = [followup]
        changes = {}
        matched_nodules = {}
        registration_quality = 0.0
        
        if self.enable_registration and self.nodule_registration:
            try:
                # 使用配准模块进行结节匹配
                matched_pairs = self.nodule_registration.register_temporal_nodules(
                    baseline_nodules, followup_nodules, registration_transform=None
                )
                
                # 处理匹配结果
                for pair in matched_pairs:
                    nodule_id = pair.baseline_nodule.nodule_id
                    matched_nodules[nodule_id] = (pair.baseline_nodule, pair.followup_nodule)
                    
                    # 使用配准模块计算的详细变化指标
                    changes[nodule_id] = pair.temporal_changes.copy()
                    changes[nodule_id]["match_score"] = pair.match_score
                    changes[nodule_id]["iou_score"] = pair.iou_score
                    changes[nodule_id]["spatial_distance"] = pair.spatial_distance
                    changes[nodule_id]["morphological_similarity"] = pair.morphological_similarity
                    
                    registration_quality = max(registration_quality, pair.match_score)
                    
            except Exception as e:
                logger.warning(f"配准失败，使用备用匹配策略: {e}")
                # 备用：使用改进的简单匹配
                self._fallback_matching(baseline_nodules, followup_nodules, matched_nodules, changes)
        else:
            # 备用匹配策略
            self._fallback_matching(baseline_nodules, followup_nodules, matched_nodules, changes)
        
        return LongitudinalPair(
            patient_id=patient_id,
            baseline_date=baseline.study_date,
            followup_date=followup.study_date,
            baseline_nodules=baseline_nodules,
            followup_nodules=followup_nodules,
            matched_nodules=matched_nodules,
            changes=changes,
            registration_quality=registration_quality
        )
    
    def _calculate_centroid_distance(self, centroid1: Tuple[float, float, float], 
                                        centroid2: Tuple[float, float, float]) -> float:
        """计算质心距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(centroid1, centroid2)))
    
    def _fallback_matching(self, baseline_nodules: List[NoduleInfo], followup_nodules: List[NoduleInfo],
                            matched_nodules: Dict[str, Tuple[NoduleInfo, NoduleInfo]], 
                            changes: Dict[str, Dict[str, float]]) -> None:
        """备用匹配策略（改进的简单匹配）"""
        for b_nodule in baseline_nodules:
            best_match = None
            best_score = 0.0
            
            for f_nodule in followup_nodules:
                # 计算多特征匹配分数
                spatial_distance = self._calculate_centroid_distance(b_nodule.centroid, f_nodule.centroid)
                volume_ratio = min(b_nodule.volume_mm3, f_nodule.volume_mm3) / max(b_nodule.volume_mm3, f_nodule.volume_mm3, 1e-6)
                density_diff = abs(b_nodule.mean_hu - f_nodule.mean_hu) / max(abs(b_nodule.mean_hu), 100)
                
                # 综合匹配分数（距离越小越好，体积和密度相似度越高越好）
                spatial_score = max(0, 1.0 - spatial_distance / 50.0)  # 50mm归一化
                volume_score = volume_ratio
                density_score = max(0, 1.0 - density_diff)
                
                total_score = (spatial_score * 0.5 + volume_score * 0.3 + density_score * 0.2)
                
                if total_score > best_score and total_score > 0.3:  # 最小匹配阈值
                    best_score = total_score
                    best_match = f_nodule
            
            if best_match:
                nodule_id = b_nodule.nodule_id
                matched_nodules[nodule_id] = (b_nodule, best_match)
                changes[nodule_id] = self._calculate_nodule_changes(b_nodule, best_match)
                changes[nodule_id]["match_score"] = best_score
                changes[nodule_id]["fallback_matching"] = True  # 标记为备用匹配
    
    def _calculate_nodule_changes(self, baseline: NoduleInfo, followup: NoduleInfo) -> Dict[str, float]:
        """计算结节变化"""
        volume_change = ((followup.volume_mm3 - baseline.volume_mm3) / max(baseline.volume_mm3, 1e-6)) * 100
        density_change = followup.mean_hu - baseline.mean_hu
        diameter_change = followup.diameter_mm - baseline.diameter_mm
        
        return {
            "volume_change_percent": volume_change,
            "density_change_hu": density_change,
            "diameter_change_mm": diameter_change,
            "volume_ratio": followup.volume_mm3 / max(baseline.volume_mm3, 1e-6)
        }
    
    def generate_instruction_label_pairs(self, longitudinal_pairs: List[LongitudinalPair], 
                                            language: str = "zh") -> List[Dict[str, Any]]:
        """生成指令-标签对"""
        samples = []
        
        for pair in longitudinal_pairs:
            # 为每个任务类型生成样本
            for task_type in self.task_types.keys():
                sample = self._create_sample(pair, task_type, language)
                if sample:
                    samples.append(sample)
        
        return samples
    
    def _create_sample(self, pair: LongitudinalPair, task_type: str, language: str) -> Optional[Dict[str, Any]]:
        """创建单个样本"""
        try:
            # 获取图像路径
            baseline_image = pair.baseline_nodules[0].image_path if pair.baseline_nodules else ""
            followup_image = pair.followup_nodules[0].image_path if pair.followup_nodules else ""
            
            # 生成目标掩码（基于任务类型）
            target_mask = self._generate_target_mask(pair, task_type)
            
            if target_mask is None:
                return None
            
            # 生成指令
            instruction = self._generate_instruction(task_type, baseline_image, followup_image, 
                                                        pair, language)
            
            # 生成变化信息
            changes = self._aggregate_changes(pair)
            
            return {
                "id": f"{pair.patient_id}_{pair.baseline_date}_{pair.followup_date}_{task_type}",
                "patient_id": pair.patient_id,
                "baseline_date": pair.baseline_date,
                "followup_date": pair.followup_date,
                "image_t0": baseline_image,
                "image_t1": followup_image,
                "target_mask": target_mask,
                "task_type": task_type,
                "instruction": instruction,
                "response": "[SEG]",  # 分割任务的标准响应
                "changes": changes,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"Error creating sample for {task_type}: {e}")
            return None
    
    def _generate_target_mask(self, pair: LongitudinalPair, task_type: str) -> Optional[str]:
        """生成目标掩码路径"""
        # 这里简化处理，实际应用中需要根据任务类型生成对应的掩码
        if task_type == "volume_threshold":
            # 选择体积变化超过阈值的结节
            for nodule_id, changes in pair.changes.items():
                if changes["volume_change_percent"] >= self.volume_threshold:
                    return pair.matched_nodules[nodule_id][1].mask_path
        
        elif task_type == "new_lesion":
            # 新发病灶（简化处理）
            if pair.followup_nodules:
                return pair.followup_nodules[0].mask_path
        
        elif task_type == "density_change":
            # 密度变化超过阈值
            for nodule_id, changes in pair.changes.items():
                if abs(changes["density_change_hu"]) >= self.density_threshold:
                    return pair.matched_nodules[nodule_id][1].mask_path
        
        elif task_type == "combined_attributes":
            # 组合条件
            for nodule_id, changes in pair.changes.items():
                if (changes["volume_change_percent"] >= self.volume_threshold and 
                    abs(changes["density_change_hu"]) >= self.density_threshold):
                    return pair.matched_nodules[nodule_id][1].mask_path
        
        return None
    
    def _generate_instruction(self, task_type: str, image_t0: str, image_t1: str, 
                                pair: LongitudinalPair, language: str) -> str:
        """生成指令"""
        template = self.instruction_templates[task_type][language]
        
        # 根据任务类型填充参数
        if task_type == "volume_threshold":
            return template.format(image_t0=image_t0, image_t1=image_t1, 
                                    threshold=self.volume_threshold)

        elif task_type == "new_lesion":
            lesion_type = "磨玻璃" if language == "zh" else "ground_glass"
            return template.format(image_t0=image_t0, image_t1=image_t1, 
                                    lesion_type=lesion_type)

        elif task_type == "density_change":
            return template.format(image_t0=image_t0, image_t1=image_t1, 
                                    threshold=self.density_threshold)

        elif task_type == "combined_attributes":
            return template.format(image_t0=image_t1, image_t1=image_t1, 
                                        vol_threshold=self.volume_threshold,
                                        den_threshold=self.density_threshold)

        return template.format(image_t0=image_t0, image_t1=image_t1)
    
    def _aggregate_changes(self, pair: LongitudinalPair) -> Dict[str, float]:
        """聚合变化信息"""
        if not pair.changes:
            return {}

        # 计算平均变化
        volume_changes = [changes["volume_change_percent"] for changes in pair.changes.values()]
        density_changes = [changes["density_change_hu"] for changes in pair.changes.values()]

        return {
            "avg_volume_change_percent": np.mean(volume_changes) if volume_changes else 0,
            "avg_density_change_hu": np.mean(density_changes) if density_changes else 0,
            "max_volume_change_percent": max(volume_changes) if volume_changes else 0,
            "max_density_change_hu": max(density_changes, key=abs) if density_changes else 0
        }

    def save_dataset(self, samples: List[Dict[str, Any]], output_file: str):
        """保存数据集"""
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_samples": len(samples),
                "task_types": list(self.task_types.keys()),
                "volume_threshold": self.volume_threshold,
                "density_threshold": self.density_threshold,
                "min_nodule_volume": self.min_nodule_volume,
                "max_followup_days": self.max_followup_days
            },
            "samples": samples
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved to {output_file}")

    def generate_dataset(self, metadata_file: str, output_file: str, language: str = "zh"):
        """完整的数据集生成流程"""
        logger.info("Loading LIDC metadata...")
        patient_nodules = self.load_lidc_data(metadata_file)

        logger.info("Finding longitudinal pairs...")
        longitudinal_pairs = self.find_longitudinal_pairs(patient_nodules)

        logger.info(f"Found {len(longitudinal_pairs)} longitudinal pairs")

        logger.info("Generating instruction-label pairs...")
        samples = self.generate_instruction_label_pairs(longitudinal_pairs, language)

        logger.info(f"Generated {len(samples)} samples")

        logger.info("Saving dataset...")
        self.save_dataset(samples, output_file)

        return samples

def create_lidc_longitudinal_dataset(data_root: str, metadata_file: str,
                                        output_file: str, **kwargs) -> List[Dict[str, Any]]:
    """便捷函数：创建LIDC纵向数据集"""
    generator = LIDCLongitudinalDataGenerator(data_root, **kwargs)
    return generator.generate_dataset(metadata_file, output_file)

if __name__ == "__main__":
    # 示例用法
    data_root = "/path/to/lidc/data"
    metadata_file = "/path/to/lidc_metadata.json"
    output_file = "/path/to/lidc_longitudinal_dataset.json"

    samples = create_lidc_longitudinal_dataset(
        data_root=data_root,
        metadata_file=metadata_file,
        output_file=output_file,
        volume_threshold=25.0,
        density_threshold=50.0,
        language="zh"
    )

    print(f"Generated {len(samples)} longitudinal samples")