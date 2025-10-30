"""
LIDC-IDRI纵向推理分割推理指令-标签对生成器
基于变化计算结果生成多样化的推理指令和对应的标签
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

from llava.data.longitudinal_changes import LongitudinalChangeMetrics, LongitudinalChangeCalculator

logger = logging.getLogger(__name__)

@dataclass
class InstructionLabelPair:
    """推理指令-标签对"""
    instruction: str
    target_mask: np.ndarray  # 目标掩膜
    instruction_type: str  # 任务类型
    change_metrics: Optional[LongitudinalChangeMetrics] = None
    reasoning_path: Optional[str] = None  # 推理路径描述
    difficulty_level: str = "medium"  # easy, medium, hard
    clinical_relevance: str = "high"  # low, medium, high
    language: str = "en"  # en, zh
    metadata: Optional[Dict[str, Any]] = None

class LongitudinalInstructionGenerator:
    """纵向推理分割指令生成器"""
    
    def __init__(
        self,
        volume_thresholds: List[float] = [20, 25, 30, 40, 50],
        density_thresholds: List[float] = [30, 50, 75, 100, 150],
        diameter_thresholds: List[float] = [1.0, 2.0, 3.0, 5.0],
        languages: List[str] = ["en", "zh"],
        instruction_complexity: str = "mixed"  # simple, mixed, complex
    ):
        self.volume_thresholds = volume_thresholds
        self.density_thresholds = density_thresholds
        self.diameter_thresholds = diameter_thresholds
        self.languages = languages
        self.instruction_complexity = instruction_complexity
        
        # 任务类型定义
        self.task_types = {
            "T1_volume_threshold": {
                "name": "Volume Threshold Reasoning",
                "description": "基于体积阈值的变化推理",
                "complexity": "medium"
            },
            "T2_new_disappeared": {
                "name": "New/Disappeared Lesions",
                "description": "新出现或消失的病灶",
                "complexity": "easy"
            },
            "T3_density_morphology": {
                "name": "Density/Morphology Changes",
                "description": "密度或形态学变化",
                "complexity": "hard"
            },
            "T4_multi_attribute": {
                "name": "Multi-Attribute Combination",
                "description": "多属性组合推理",
                "complexity": "hard"
            },
            "T5_temporal_progression": {
                "name": "Temporal Progression Assessment",
                "description": "时间进展评估",
                "complexity": "hard"
            }
        }
        
        # 指令模板
        self.instruction_templates = self._load_instruction_templates()
        
        # 推理路径模板
        self.reasoning_templates = self._load_reasoning_templates()
        
        logger.info(f"Initialized LongitudinalInstructionGenerator with {len(self.task_types)} task types")
    
    def _load_instruction_templates(self) -> Dict[str, List[str]]:
        """加载指令模板"""
        return {
            "en": {
                "T1_volume_threshold": [
                    "Segment all nodules that have increased in volume by ≥{threshold}% compared to the previous scan.",
                    "Identify nodules with volume growth exceeding {threshold}% from baseline.",
                    "Highlight lesions that show {threshold}% or more volumetric progression.",
                    "Find all nodules where the volume has expanded by at least {threshold}%.",
                    "Segment nodules demonstrating significant volume increase (≥{threshold}%)."
                ],
                "T2_new_disappeared": [
                    "Segment all newly appeared nodules in the current scan.",
                    "Identify nodules that were not present in the previous examination.",
                    "Highlight new lesions that emerged since the last scan.",
                    "Find all ground-glass nodules that appeared in this follow-up.",
                    "Segment disappeared nodules that were present before but absent now."
                ],
                "T3_density_morphology": [
                    "Segment nodules that changed from ground-glass to solid appearance.",
                    "Identify lesions with increased density (≥{threshold} HU change).",
                    "Highlight nodules showing morphological changes in shape or texture.",
                    "Find nodules with blurred margins compared to previous scan.",
                    "Segment lesions demonstrating density progression of {threshold} HU or more."
                ],
                "T4_multi_attribute": [
                    "Segment nodules with volume increase ≥{vol_threshold}% AND density increase ≥{dens_threshold} HU.",
                    "Identify lesions showing both volumetric and morphological progression.",
                    "Highlight nodules with diameter growth ≥{diam_threshold}mm AND density change ≥{dens_threshold} HU.",
                    "Find nodules meeting multiple progression criteria simultaneously.",
                    "Segment lesions with combined volume and spatial changes."
                ],
                "T5_temporal_progression": [
                    "Assess temporal progression of all nodules and segment high-risk cases.",
                    "Identify nodules requiring immediate follow-up based on progression patterns.",
                    "Segment lesions showing concerning temporal evolution characteristics.",
                    "Highlight nodules with rapid progression indicating clinical significance.",
                    "Find all nodules demonstrating temporal changes warranting intervention."
                ]
            },
            "zh": {
                "T1_volume_threshold": [
                    "分割所有与上次扫描相比体积增加≥{threshold}%的结节。",
                    "识别体积增长超过基线{threshold}%的结节。",
                    "突出显示体积进展达到{threshold}%或以上的病变。",
                    "找出体积至少扩大{threshold}%的所有结节。",
                    "分割显示显著体积增加（≥{threshold}%）的结节。"
                ],
                "T2_new_disappeared": [
                    "分割当前扫描中新出现的所有结节。",
                    "识别之前检查中不存在的结节。",
                    "突出显示自上次扫描以来新出现的病变。",
                    "找出本次随访中出现的所有磨玻璃结节。",
                    "分割之前存在但现在消失的结节。"
                ],
                "T3_density_morphology": [
                    "分割从磨玻璃外观变为实性的结节。",
                    "识别密度增加（≥{threshold} HU变化）的病变。",
                    "突出显示形状或纹理发生形态学变化的结节。",
                    "找出与之前扫描相比边界模糊的结节。",
                    "分割显示{threshold} HU或更多密度进展的病变。"
                ],
                "T4_multi_attribute": [
                    "分割体积增加≥{vol_threshold}% AND密度增加≥{dens_threshold} HU的结节。",
                    "识别同时显示体积和形态进展的病变。",
                    "突出显示直径增长≥{diam_threshold}mm AND密度变化≥{dens_threshold} HU的结节。",
                    "找出同时满足多个进展标准的结节。",
                    "分割具有组合体积和空间变化的病变。"
                ],
                "T5_temporal_progression": [
                    "评估所有结节的时间进展并分割高风险病例。",
                    "基于进展模式识别需要立即随访的结节。",
                    "分割显示令人担忧的时间演化特征的病变。",
                    "突出显示快速进展表明临床意义的结节。",
                    "找出所有显示需要干预的时间变化的结节。"
                ]
            }
        }
    
    def _load_reasoning_templates(self) -> Dict[str, Dict[str, str]]:
        """加载推理路径模板"""
        return {
            "T1_volume_threshold": {
                "reasoning": "Volume change calculation → Threshold comparison → Segmentation",
                "explanation": "The model calculates volume changes between timepoints and segments nodules exceeding the specified threshold."
            },
            "T2_new_disappeared": {
                "reasoning": "Temporal comparison → New lesion detection → Segmentation",
                "explanation": "The model compares current and previous scans to identify newly appeared or disappeared lesions."
            },
            "T3_density_morphology": {
                "reasoning": "Density analysis → Morphological assessment → Change detection → Segmentation",
                "explanation": "The model analyzes density and morphological changes to identify transformed lesions."
            },
            "T4_multi_attribute": {
                "reasoning": "Multi-feature analysis → Combined thresholding → Segmentation",
                "explanation": "The model evaluates multiple attributes simultaneously to identify lesions meeting all criteria."
            },
            "T5_temporal_progression": {
                "reasoning": "Temporal pattern analysis → Risk assessment → Clinical significance → Segmentation",
                "explanation": "The model assesses temporal progression patterns to identify clinically significant changes."
            }
        }
    
    def generate_instruction_label_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        nodule_id: str = "unknown",
        num_pairs_per_type: int = 3,
        include_negative_samples: bool = True,
        randomize_thresholds: bool = True
    ) -> List[InstructionLabelPair]:
        """生成指令-标签对"""
        
        instruction_pairs = []
        
        # 为每种任务类型生成指令
        for task_type, task_info in self.task_types.items():
            if task_info["complexity"] == "easy" and self.instruction_complexity == "complex":
                continue
            if task_info["complexity"] == "hard" and self.instruction_complexity == "simple":
                continue
            
            # 生成该类型的指令对
            type_pairs = self._generate_type_specific_pairs(
                task_type, change_metrics, mask_t0, mask_t1, ct_t0, ct_t1,
                num_pairs=num_pairs_per_type,
                randomize_thresholds=randomize_thresholds,
                include_negative=include_negative_samples
            )
            
            instruction_pairs.extend(type_pairs)
        
        logger.info(f"Generated {len(instruction_pairs)} instruction-label pairs for nodule {nodule_id}")
        return instruction_pairs
    
    def _generate_type_specific_pairs(
        self,
        task_type: str,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        num_pairs: int = 3,
        randomize_thresholds: bool = True,
        include_negative: bool = True
    ) -> List[InstructionLabelPair]:
        """生成特定类型的指令对"""
        
        pairs = []
        
        if task_type == "T1_volume_threshold":
            pairs = self._generate_volume_threshold_pairs(
                change_metrics, mask_t0, mask_t1, num_pairs, randomize_thresholds, include_negative
            )
        elif task_type == "T2_new_disappeared":
            pairs = self._generate_new_disappeared_pairs(
                change_metrics, mask_t0, mask_t1, num_pairs, include_negative
            )
        elif task_type == "T3_density_morphology":
            pairs = self._generate_density_morphology_pairs(
                change_metrics, mask_t0, mask_t1, ct_t0, ct_t1, num_pairs, randomize_thresholds, include_negative
            )
        elif task_type == "T4_multi_attribute":
            pairs = self._generate_multi_attribute_pairs(
                change_metrics, mask_t0, mask_t1, ct_t0, ct_t1, num_pairs, randomize_thresholds, include_negative
            )
        elif task_type == "T5_temporal_progression":
            pairs = self._generate_temporal_progression_pairs(
                change_metrics, mask_t0, mask_t1, num_pairs, include_negative
            )
        
        return pairs
    
    def _generate_volume_threshold_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        num_pairs: int,
        randomize_thresholds: bool,
        include_negative: bool
    ) -> List[InstructionLabelPair]:
        """生成体积阈值指令对"""
        
        pairs = []
        actual_volume_change = change_metrics.volume_change_percent
        
        # 选择阈值
        thresholds = self.volume_thresholds.copy()
        if randomize_thresholds:
            random.shuffle(thresholds)
        
        used_thresholds = set()
        
        for threshold in thresholds[:num_pairs]:
            if threshold in used_thresholds:
                continue
            used_thresholds.add(threshold)
            
            # 判断是否符合条件
            meets_criteria = actual_volume_change >= threshold
            
            # 生成目标掩膜
            target_mask = mask_t1 if meets_criteria else np.zeros_like(mask_t1)
            
            # 生成正样本指令
            for lang in self.languages:
                templates = self.instruction_templates[lang]["T1_volume_threshold"]
                template = random.choice(templates)
                instruction = template.format(threshold=threshold)
                
                pair = InstructionLabelPair(
                    instruction=instruction,
                    target_mask=target_mask,
                    instruction_type="T1_volume_threshold",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T1_volume_threshold"]["reasoning"],
                    difficulty_level="medium",
                    clinical_relevance="high",
                    language=lang,
                    metadata={
                        "threshold": threshold,
                        "actual_change": actual_volume_change,
                        "meets_criteria": meets_criteria,
                        "task_subtype": "volume_progression"
                    }
                )
                pairs.append(pair)
            
            # 生成负样本指令（反向条件）
            if include_negative:
                for lang in self.languages:
                    neg_templates = [
                        f"Segment all nodules that have NOT increased in volume by ≥{threshold}%.",
                        f"Identify nodules with volume change less than {threshold}%.",
                        f"Find nodules showing stable or decreased volume (<{threshold}% increase)."
                    ]
                    neg_template = random.choice(neg_templates)
                    
                    # 负样本的目标掩膜是相反的情况
                    neg_target_mask = np.zeros_like(mask_t1) if meets_criteria else mask_t1
                    
                    neg_pair = InstructionLabelPair(
                        instruction=neg_template,
                        target_mask=neg_target_mask,
                        instruction_type="T1_volume_threshold",
                        change_metrics=change_metrics,
                        reasoning_path=self.reasoning_templates["T1_volume_threshold"]["reasoning"],
                        difficulty_level="medium",
                        clinical_relevance="medium",
                        language=lang,
                        metadata={
                            "threshold": threshold,
                            "actual_change": actual_volume_change,
                            "meets_criteria": not meets_criteria,
                            "task_subtype": "volume_stability",
                            "negative_sample": True
                        }
                    )
                    pairs.append(neg_pair)
        
        return pairs
    
    def _generate_new_disappeared_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        num_pairs: int,
        include_negative: bool
    ) -> List[InstructionLabelPair]:
        """生成新出现/消失病灶指令对"""
        
        pairs = []
        
        # 判断是否有新病灶（基于空间重叠度）
        spatial_overlap = change_metrics.spatial_overlap_iou
        has_new_lesions = spatial_overlap < 0.3  # IoU小于0.3认为有新病灶
        has_disappeared = spatial_overlap < 0.3 and change_metrics.volume_change_percent < -50
        
        # 生成新病灶指令
        new_templates = self.instruction_templates["en"]["T2_new_disappeared"]
        for template in random.sample(new_templates, min(num_pairs, len(new_templates))):
            instruction = template
            target_mask = mask_t1 if has_new_lesions else np.zeros_like(mask_t1)
            
            pair = InstructionLabelPair(
                instruction=instruction,
                target_mask=target_mask,
                instruction_type="T2_new_disappeared",
                change_metrics=change_metrics,
                reasoning_path=self.reasoning_templates["T2_new_disappeared"]["reasoning"],
                difficulty_level="easy",
                clinical_relevance="high",
                language="en",
                metadata={
                    "has_new_lesions": has_new_lesions,
                    "spatial_overlap": spatial_overlap,
                    "task_subtype": "new_lesions"
                }
            )
            pairs.append(pair)
        
        # 生成消失病灶指令
        if has_disappeared:
            disappeared_templates = [
                "Segment nodules that disappeared compared to the previous scan.",
                "Identify lesions that were present before but are now absent.",
                "Find disappeared nodules from baseline examination."
            ]
            
            for template in random.sample(disappeared_templates, min(num_pairs, len(disappeared_templates))):
                pair = InstructionLabelPair(
                    instruction=template,
                    target_mask=np.zeros_like(mask_t1),  # 消失病灶的目标掩膜为空
                    instruction_type="T2_new_disappeared",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T2_new_disappeared"]["reasoning"],
                    difficulty_level="easy",
                    clinical_relevance="high",
                    language="en",
                    metadata={
                        "has_disappeared": has_disappeared,
                        "task_subtype": "disappeared_lesions"
                    }
                )
                pairs.append(pair)
        
        return pairs
    
    def _generate_density_morphology_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        num_pairs: int,
        randomize_thresholds: bool,
        include_negative: bool
    ) -> List[InstructionLabelPair]:
        """生成密度形态学变化指令对"""
        
        pairs = []
        actual_density_change = change_metrics.density_change_hu
        
        # 选择阈值
        thresholds = self.density_thresholds.copy()
        if randomize_thresholds:
            random.shuffle(thresholds)
        
        # 密度变化指令
        for threshold in thresholds[:num_pairs]:
            meets_criteria = actual_density_change >= threshold
            target_mask = mask_t1 if meets_criteria else np.zeros_like(mask_t1)
            
            templates = self.instruction_templates["en"]["T3_density_morphology"]
            template = random.choice(templates)
            instruction = template.format(threshold=threshold)
            
            pair = InstructionLabelPair(
                instruction=instruction,
                target_mask=target_mask,
                instruction_type="T3_density_morphology",
                change_metrics=change_metrics,
                reasoning_path=self.reasoning_templates["T3_density_morphology"]["reasoning"],
                difficulty_level="hard",
                clinical_relevance="high",
                language="en",
                metadata={
                    "threshold": threshold,
                    "actual_density_change": actual_density_change,
                    "meets_criteria": meets_criteria,
                    "task_subtype": "density_progression"
                }
            )
            pairs.append(pair)
        
        # 形态学变化指令
        if abs(change_metrics.sphericity_change) > 0.1:  # 显著的球形度变化
            morphology_templates = [
                "Segment nodules with significant shape changes.",
                "Identify lesions showing morphological evolution.",
                "Highlight nodules with altered sphericity or compactness."
            ]
            
            for template in random.sample(morphology_templates, min(2, len(morphology_templates))):
                meets_criteria = abs(change_metrics.sphericity_change) > 0.1
                target_mask = mask_t1 if meets_criteria else np.zeros_like(mask_t1)
                
                pair = InstructionLabelPair(
                    instruction=template,
                    target_mask=target_mask,
                    instruction_type="T3_density_morphology",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T3_density_morphology"]["reasoning"],
                    difficulty_level="hard",
                    clinical_relevance="medium",
                    language="en",
                    metadata={
                        "sphericity_change": change_metrics.sphericity_change,
                        "meets_criteria": meets_criteria,
                        "task_subtype": "morphology_change"
                    }
                )
                pairs.append(pair)
        
        return pairs
    
    def _generate_multi_attribute_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        num_pairs: int,
        randomize_thresholds: bool,
        include_negative: bool
    ) -> List[InstructionLabelPair]:
        """生成多属性组合指令对"""
        
        pairs = []
        
        # 组合不同的阈值
        vol_thresholds = random.sample(self.volume_thresholds, min(2, len(self.volume_thresholds)))
        dens_thresholds = random.sample(self.density_thresholds, min(2, len(self.density_thresholds)))
        diam_thresholds = random.sample(self.diameter_thresholds, min(2, len(self.diameter_thresholds)))
        
        # 生成组合条件
        for vol_thresh in vol_thresholds[:num_pairs]:
            for dens_thresh in dens_thresholds[:1]:  # 每个体积阈值只配一个密度阈值
                meets_volume = change_metrics.volume_change_percent >= vol_thresh
                meets_density = change_metrics.density_change_hu >= dens_thresh
                meets_both = meets_volume and meets_density
                
                target_mask = mask_t1 if meets_both else np.zeros_like(mask_t1)
                
                templates = self.instruction_templates["en"]["T4_multi_attribute"]
                template = random.choice(templates)
                instruction = template.format(
                    vol_threshold=vol_thresh,
                    dens_threshold=dens_thresh,
                    diam_threshold=diam_thresholds[0]
                )
                
                pair = InstructionLabelPair(
                    instruction=instruction,
                    target_mask=target_mask,
                    instruction_type="T4_multi_attribute",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T4_multi_attribute"]["reasoning"],
                    difficulty_level="hard",
                    clinical_relevance="high",
                    language="en",
                    metadata={
                        "volume_threshold": vol_thresh,
                        "density_threshold": dens_thresh,
                        "meets_volume": meets_volume,
                        "meets_density": meets_density,
                        "meets_both": meets_both,
                        "task_subtype": "volume_density_combination"
                    }
                )
                pairs.append(pair)
        
        return pairs
    
    def _generate_temporal_progression_pairs(
        self,
        change_metrics: LongitudinalChangeMetrics,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray,
        num_pairs: int,
        include_negative: bool
    ) -> List[InstructionLabelPair]:
        """生成时间进展评估指令对"""
        
        pairs = []
        
        # 基于综合进展评分
        progression_score = change_metrics.overall_progression_score
        
        # 高风险进展
        if progression_score > 0.7:
            templates = self.instruction_templates["en"]["T5_temporal_progression"]
            for template in random.sample(templates, min(num_pairs, len(templates))):
                pair = InstructionLabelPair(
                    instruction=template,
                    target_mask=mask_t1,  # 高风险病例分割当前掩膜
                    instruction_type="T5_temporal_progression",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T5_temporal_progression"]["reasoning"],
                    difficulty_level="hard",
                    clinical_relevance="high",
                    language="en",
                    metadata={
                        "progression_score": progression_score,
                        "risk_level": "high",
                        "task_subtype": "high_risk_progression"
                    }
                )
                pairs.append(pair)
        
        # 稳定病例
        elif progression_score < 0.3:
            stable_templates = [
                "Identify nodules showing stable characteristics over time.",
                "Segment lesions with no significant temporal changes.",
                "Highlight nodules demonstrating temporal stability."
            ]
            
            for template in random.sample(stable_templates, min(num_pairs, len(stable_templates))):
                pair = InstructionLabelPair(
                    instruction=template,
                    target_mask=mask_t1,  # 稳定病例也分割当前掩膜
                    instruction_type="T5_temporal_progression",
                    change_metrics=change_metrics,
                    reasoning_path=self.reasoning_templates["T5_temporal_progression"]["reasoning"],
                    difficulty_level="hard",
                    clinical_relevance="medium",
                    language="en",
                    metadata={
                        "progression_score": progression_score,
                        "risk_level": "low",
                        "task_subtype": "stable_disease"
                    }
                )
                pairs.append(pair)
        
        return pairs
    
    def generate_dataset(
        self,
        longitudinal_data: List[Dict[str, Any]],
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        save_format: str = "json",
        include_metadata: bool = True
    ) -> Dict[str, List[InstructionLabelPair]]:
        """生成完整的数据集"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_pairs = []
        
        # 处理每个纵向数据样本
        for i, data_item in enumerate(longitudinal_data):
            logger.info(f"Processing sample {i+1}/{len(longitudinal_data)}")
            
            # 提取数据
            mask_t0 = data_item["mask_t0"]
            mask_t1 = data_item["mask_t1"]
            ct_t0 = data_item["ct_t0"]
            ct_t1 = data_item["ct_t1"]
            nodule_id = data_item.get("nodule_id", f"nodule_{i}")
            
            # 计算变化指标
            calculator = LongitudinalChangeCalculator()
            change_metrics = calculator.compute_longitudinal_changes(
                mask_t0, mask_t1, ct_t0, ct_t1, nodule_id=nodule_id
            )
            
            # 生成指令-标签对
            pairs = self.generate_instruction_label_pairs(
                change_metrics, mask_t0, mask_t1, ct_t0, ct_t1,
                nodule_id=nodule_id,
                num_pairs_per_type=3,
                include_negative_samples=True
            )
            
            all_pairs.extend(pairs)
        
        logger.info(f"Generated total {len(all_pairs)} instruction-label pairs")
        
        # 数据分割
        random.shuffle(all_pairs)
        total_samples = len(all_pairs)
        train_end = int(total_samples * split_ratios[0])
        val_end = train_end + int(total_samples * split_ratios[1])
        
        splits = {
            "train": all_pairs[:train_end],
            "val": all_pairs[train_end:val_end],
            "test": all_pairs[val_end:]
        }
        
        # 保存数据
        for split_name, split_data in splits.items():
            split_path = output_path / f"{split_name}.{save_format}"
            self._save_split_data(split_data, split_path, save_format, include_metadata)
            logger.info(f"Saved {len(split_data)} samples to {split_path}")
        
        # 保存统计信息
        stats = self._compute_dataset_stats(all_pairs)
        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return splits
    
    def _save_split_data(
        self,
        data: List[InstructionLabelPair],
        output_path: Path,
        save_format: str,
        include_metadata: bool
    ):
        """保存分割数据"""
        
        if save_format == "json":
            # 转换为可序列化格式
            serializable_data = []
            for pair in data:
                item = {
                    "instruction": pair.instruction,
                    "target_mask": pair.target_mask.tolist() if isinstance(pair.target_mask, np.ndarray) else pair.target_mask,
                    "instruction_type": pair.instruction_type,
                    "difficulty_level": pair.difficulty_level,
                    "clinical_relevance": pair.clinical_relevance,
                    "language": pair.language,
                    "reasoning_path": pair.reasoning_path
                }
                
                if include_metadata and pair.metadata:
                    item["metadata"] = pair.metadata
                
                if pair.change_metrics:
                    item["change_metrics"] = asdict(pair.change_metrics)
                
                serializable_data.append(item)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        elif save_format == "npz":
            # 保存为numpy格式
            instructions = [pair.instruction for pair in data]
            target_masks = np.array([pair.target_mask for pair in data])
            instruction_types = [pair.instruction_type for pair in data]
            
            np.savez_compressed(
                output_path,
                instructions=instructions,
                target_masks=target_masks,
                instruction_types=instruction_types
            )
    
    def _compute_dataset_stats(self, data: List[InstructionLabelPair]) -> Dict[str, Any]:
        """计算数据集统计信息"""
        
        stats = {
            "total_samples": len(data),
            "task_type_distribution": {},
            "language_distribution": {},
            "difficulty_distribution": {},
            "clinical_relevance_distribution": {},
            "instruction_length_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": float('inf'),
                "max": 0
            }
        }
        
        instruction_lengths = []
        
        for pair in data:
            # 任务类型分布
            task_type = pair.instruction_type
            stats["task_type_distribution"][task_type] = stats["task_type_distribution"].get(task_type, 0) + 1
            
            # 语言分布
            lang = pair.language
            stats["language_distribution"][lang] = stats["language_distribution"].get(lang, 0) + 1
            
            # 难度分布
            difficulty = pair.difficulty_level
            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
            
            # 临床相关性分布
            relevance = pair.clinical_relevance
            stats["clinical_relevance_distribution"][relevance] = stats["clinical_relevance_distribution"].get(relevance, 0) + 1
            
            # 指令长度统计
            instruction_lengths.append(len(pair.instruction))
        
        # 计算指令长度统计
        if instruction_lengths:
            stats["instruction_length_stats"]["mean"] = np.mean(instruction_lengths)
            stats["instruction_length_stats"]["std"] = np.std(instruction_lengths)
            stats["instruction_length_stats"]["min"] = min(instruction_lengths)
            stats["instruction_length_stats"]["max"] = max(instruction_lengths)
        
        return stats

# 便捷函数
def create_longitudinal_instruction_dataset(
    longitudinal_data: List[Dict[str, Any]],
    output_dir: str,
    **kwargs
) -> Dict[str, List[InstructionLabelPair]]:
    """便捷函数：创建纵向推理指令数据集"""
    generator = LongitudinalInstructionGenerator(**kwargs)
    return generator.generate_dataset(longitudinal_data, output_dir)

if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例数据
    sample_data = []
    for i in range(5):
        mask_t0 = np.random.rand(32, 32, 16) > 0.8
        mask_t1 = np.random.rand(32, 32, 16) > 0.7
        ct_t0 = np.random.randint(-1000, 400, (32, 32, 16))
        ct_t1 = np.random.randint(-1000, 400, (32, 32, 16))
        
        sample_data.append({
            "mask_t0": mask_t0,
            "mask_t1": mask_t1,
            "ct_t0": ct_t0,
            "ct_t1": ct_t1,
            "nodule_id": f"sample_{i}"
        })
    
    # 生成数据集
    output_dir = "longitudinal_instruction_dataset"
    splits = create_longitudinal_instruction_dataset(sample_data, output_dir)
    
    print(f"Dataset created with {len(splits['train'])} training samples")
    print(f"Dataset created with {len(splits['val'])} validation samples")
    print(f"Dataset created with {len(splits['test'])} test samples")