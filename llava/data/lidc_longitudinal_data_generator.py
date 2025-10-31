"""
LIDC-IDRI纵向推理分割数据生成器
基于现有SegLLM架构，实现LIDC-IDRI数据集的纵向推理分割任务

主要功能：
- 加载LIDC-IDRI元数据并解析结节信息
- 查找纵向配对（baseline和follow-up扫描）
- 执行结节配准和变化检测
- 生成多任务类型的指令-标签对
- 支持体积变化、密度变化、新发病灶等任务
"""
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

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

# 配置常量
DEFAULT_VOLUME_THRESHOLD = 25.0  # 默认体积变化阈值(%)
DEFAULT_DENSITY_THRESHOLD = 50.0  # 默认密度变化阈值(HU)
DEFAULT_MIN_NODULE_VOLUME = 30.0  # 默认最小结节体积(mm³)
DEFAULT_MAX_FOLLOWUP_DAYS = 730  # 默认最大随访天数(2年)
DEFAULT_SEED = 42  # 默认随机种子
DEFAULT_IOU_THRESHOLD = 0.3  # 默认IOU阈值
DEFAULT_DISTANCE_THRESHOLD = 30.0  # 默认距离阈值(mm)
DEFAULT_MORPHOLOGICAL_WEIGHT = 0.3  # 默认形态学权重
DEFAULT_TEMPORAL_WEIGHT = 0.2  # 默认时相权重
IMAGE_TEMPLATE_SIZE = 256  # 图像模板大小
MATCH_SCORE_THRESHOLD = 0.3  # 匹配分数阈值
SPATIAL_DISTANCE_NORM = 50.0  # 空间距离归一化因子

def load_longitudinal_config(config_file: str) -> Dict[str, Any]:
    """从配置文件加载纵向数据生成器配置
    
    支持JSON和YAML格式，YAML格式支持注释
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置参数字典
    """
    try:
        config_path = Path(config_file)
        
        # 根据文件扩展名选择加载方式
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            # YAML格式 - 支持注释
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 默认使用YAML格式
            logger.warning(f"Unknown config file format: {config_path.suffix}, trying YAML")
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # 提取数据生成器相关配置
        data_config = config.get('data', {})
        generator_config = data_config.get('longitudinal_data_generator', {})
        
        # 如果没有找到专用配置，尝试从其他部分提取
        if not generator_config:
            # 从longitudinal_tasks中提取相关阈值
            tasks_config = config.get('longitudinal_tasks', {})
            change_thresholds = tasks_config.get('change_thresholds', {})
            
            generator_config = {
                'density_threshold': change_thresholds.get('density_change', DEFAULT_DENSITY_THRESHOLD),
                'min_nodule_volume': 30.0,  # 使用默认值
                'max_followup_days': DEFAULT_MAX_FOLLOWUP_DAYS,
                'seed': DEFAULT_SEED,
                'enable_registration': True,
                'registration_config': {
                    'iou_threshold': DEFAULT_IOU_THRESHOLD,
                    'distance_threshold': DEFAULT_DISTANCE_THRESHOLD,
                    'morphological_weight': DEFAULT_MORPHOLOGICAL_WEIGHT,
                    'temporal_weight': DEFAULT_TEMPORAL_WEIGHT
                },
                'image_template_size': IMAGE_TEMPLATE_SIZE,
                'match_score_threshold': MATCH_SCORE_THRESHOLD,
                'spatial_distance_norm': SPATIAL_DISTANCE_NORM
            }
            
            logger.info("Using default longitudinal data generator configuration")
        
        # 验证配置完整性
        required_keys = [
            'density_threshold', 'min_nodule_volume', 'max_followup_days',
            'image_template_size', 'match_score_threshold'
        ]
        
        for key in required_keys:
            if key not in generator_config:
                logger.warning(f"Missing configuration key '{key}', using default value")
                # 使用默认值
                defaults = {
                    'density_threshold': DEFAULT_DENSITY_THRESHOLD,
                    'min_nodule_volume': DEFAULT_MIN_NODULE_VOLUME,
                    'max_followup_days': DEFAULT_MAX_FOLLOWUP_DAYS,
                    'image_template_size': IMAGE_TEMPLATE_SIZE,
                    'match_score_threshold': MATCH_SCORE_THRESHOLD,
                    'spatial_distance_norm': SPATIAL_DISTANCE_NORM,
                    'seed': DEFAULT_SEED,
                    'enable_registration': True
                }
                generator_config[key] = defaults.get(key, 0.0)
        
        logger.info(f"Loaded longitudinal data generator configuration: {generator_config}")
        return generator_config
        
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid format in configuration file {config_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {e}")
        raise

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
        config_file: Optional[str] = None,  # 配置文件路径
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,  # 体积变化阈值(%)
        density_threshold: float = DEFAULT_DENSITY_THRESHOLD,  # 密度变化阈值(HU)
        min_nodule_volume: float = DEFAULT_MIN_NODULE_VOLUME,  # 最小结节体积(mm³)
        max_followup_days: int = DEFAULT_MAX_FOLLOWUP_DAYS,  # 最大随访天数(2年)
        enable_registration: bool = True,  # 启用配准
        registration_config: Optional[Dict[str, Any]] = None,  # 配准配置
        seed: int = DEFAULT_SEED,
        **kwargs  # 支持其他配置参数
    ):
        # 如果提供了配置文件，优先使用配置文件中的参数
        if config_file:
            try:
                config_params = load_longitudinal_config(config_file)
                logger.info(f"Loading configuration from {config_file}")
                
                # 使用配置文件中的参数，如果没有提供显式参数
                density_threshold = config_params.get('density_threshold', density_threshold)
                min_nodule_volume = config_params.get('min_nodule_volume', min_nodule_volume)
                max_followup_days = config_params.get('max_followup_days', max_followup_days)
                seed = config_params.get('seed', seed)
                enable_registration = config_params.get('enable_registration', enable_registration)
                
                # 更新全局常量（用于模板等）
                global IMAGE_TEMPLATE_SIZE, MATCH_SCORE_THRESHOLD
                IMAGE_TEMPLATE_SIZE = config_params.get('image_template_size', IMAGE_TEMPLATE_SIZE)
                MATCH_SCORE_THRESHOLD = config_params.get('match_score_threshold', MATCH_SCORE_THRESHOLD)
                
                # 处理配准配置
                if enable_registration and 'registration_config' in config_params:
                    registration_config = registration_config or {}
                    reg_config = config_params['registration_config']
                    registration_config.update({
                        'iou_threshold': reg_config.get('iou_threshold', DEFAULT_IOU_THRESHOLD),
                        'distance_threshold': reg_config.get('distance_threshold', DEFAULT_DISTANCE_THRESHOLD),
                        'morphological_weight': reg_config.get('morphological_weight', DEFAULT_MORPHOLOGICAL_WEIGHT),
                        'temporal_weight': reg_config.get('temporal_weight', DEFAULT_TEMPORAL_WEIGHT)
                    })
                    
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")
                logger.warning("Using default configuration values")
        
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
                iou_threshold=reg_config.get('iou_threshold', DEFAULT_IOU_THRESHOLD),
                distance_threshold=reg_config.get('distance_threshold', DEFAULT_DISTANCE_THRESHOLD),
                morphological_weight=reg_config.get('morphological_weight', DEFAULT_MORPHOLOGICAL_WEIGHT),
                temporal_weight=reg_config.get('temporal_weight', DEFAULT_TEMPORAL_WEIGHT)
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
                "zh": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] 分割所有较上次体积增加超过{{threshold}}%的结节",
                "en": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] Segment all nodules with volume increased by more than {{threshold}}% compared to previous scan"
            },
            "new_lesion": {
                "zh": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] 标出新出现的{{lesion_type}}病灶",
                "en": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] Identify newly appeared {{lesion_type}} lesions"
            },
            "density_change": {
                "zh": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] 分割密度变化超过{{threshold}}HU的结节",
                "en": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] Segment nodules with density change exceeding {{threshold}} HU"
            },
            "combined_attributes": {
                "zh": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] 分割体积增加≥{{vol_threshold}}%且密度变化≥{{den_threshold}}HU的结节",
                "en": f"[IMAGE{IMAGE_TEMPLATE_SIZE}:{{image_t0}}|{{image_t1}}] Segment nodules with volume increase ≥{{vol_threshold}}% and density change ≥{{den_threshold}} HU"
            }
        }
    
    def load_lidc_data(self, metadata_file: str) -> Dict[str, List[NoduleInfo]]:
        """加载LIDC元数据"""
        if not Path(metadata_file).exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML format in metadata file {metadata_file}: {e}")
            raise ValueError(f"Invalid YAML format in metadata file: {e}")
        except Exception as e:
            logger.error(f"Error loading metadata file {metadata_file}: {e}")
            raise
        
        patient_nodules = {}
        
        # 验证元数据结构
        if not isinstance(metadata, list):
            logger.error(f"Expected metadata to be a list, got {type(metadata)}")
            raise ValueError("Metadata must be a list of dictionaries")
        
        logger.info(f"Processing {len(metadata)} metadata entries")
        
        for i, item in enumerate(metadata):
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dictionary item at index {i}: {type(item)}")
                continue
            
            # 验证必需字段
            required_fields = ['patient_id', 'nodule_id', 'study_date', 'series_uid', 'mask_path', 'image_path']
            missing_fields = [field for field in required_fields if field not in item or not item[field]]
            
            if missing_fields:
                logger.warning(f"Skipping item at index {i} due to missing fields: {missing_fields}")
                continue
            
            patient_id = item['patient_id']
            study_date = item['study_date']
            
            # 验证数值字段的有效性
            volume_mm3 = float(item.get('volume_mm3', 0.0))
            mean_hu = float(item.get('mean_hu', 0.0))
            diameter_mm = float(item.get('diameter_mm', 0.0))
            
            if volume_mm3 <= 0:
                logger.warning(f"Invalid volume for nodule {item['nodule_id']}: {volume_mm3}")
                continue
            
            centroid = item.get('centroid', [0, 0, 0])
            if not isinstance(centroid, (list, tuple)) or len(centroid) != 3:
                logger.warning(f"Invalid centroid format for nodule {item['nodule_id']}: {centroid}")
                centroid = [0, 0, 0]
            
            nodule_info = NoduleInfo(
                nodule_id=item['nodule_id'],
                patient_id=patient_id,
                study_date=study_date,
                series_uid=item['series_uid'],
                mask_path=item['mask_path'],
                image_path=item['image_path'],
                volume_mm3=volume_mm3,
                mean_hu=mean_hu,
                diameter_mm=diameter_mm,
                centroid=tuple(centroid),
                lesion_type=item.get('lesion_type', 'solid'),
                malignancy_score=item.get('malignancy_score')
            )
            
            if patient_id not in patient_nodules:
                patient_nodules[patient_id] = []
            patient_nodules[patient_id].append(nodule_info)
        
        logger.info(f"Loaded data for {len(patient_nodules)} patients")
        return patient_nodules
        
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
            # 验证日期格式
            if not date1 or not date2:
                logger.warning(f"Empty date string: date1={date1}, date2={date2}")
                return 0
            
            # 验证日期格式是否为YYYYMMDD
            if len(date1) != 8 or len(date2) != 8:
                logger.warning(f"Invalid date format: date1={date1}, date2={date2}, expected YYYYMMDD")
                return 0
            
            d1 = datetime.strptime(date1, "%Y%m%d")
            d2 = datetime.strptime(date2, "%Y%m%d")
            days_diff = abs((d2 - d1).days)
            
            # 验证日期差的合理性（0-10年）
            if days_diff > 3650:
                logger.warning(f"Unusually large date difference: {days_diff} days between {date1} and {date2}")
            
            return days_diff
            
        except ValueError as e:
            logger.error(f"Date parsing error for date1={date1}, date2={date2}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error calculating date difference: {e}")
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
                logger.info(f"配准失败详情 - 患者: {patient_id}, 基线日期: {baseline.study_date}, 随访日期: {followup.study_date}")
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
        """计算质心距离
        
        Args:
            centroid1: 第一个质心的坐标 (x, y, z)
            centroid2: 第二个质心的坐标 (x, y, z)
            
        Returns:
            两个质心之间的欧几里得距离
            
        Raises:
            ValueError: 如果输入的坐标格式无效
        """
        # 验证输入参数
        if not all(isinstance(coord, (int, float)) for coord in centroid1 + centroid2):
            logger.error(f"Invalid coordinate types in centroid1={centroid1} or centroid2={centroid2}")
            raise ValueError("All coordinates must be numeric")
        
        if len(centroid1) != 3 or len(centroid2) != 3:
            logger.error(f"Invalid coordinate dimensions: centroid1={len(centroid1)}, centroid2={len(centroid2)}")
            raise ValueError("Coordinates must be 3D (x, y, z)")
        
        try:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(centroid1, centroid2)))
            
            # 验证计算结果的合理性
            if distance < 0:
                logger.error(f"Negative distance calculated: {distance}")
                return 0.0
            
            if distance > 1000:  # 超过1米的距离在医学影像中不合理
                logger.warning(f"Unusually large distance detected: {distance:.2f} between {centroid1} and {centroid2}")
            
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating centroid distance: {e}")
            raise
        """计算质心距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(centroid1, centroid2)))
    
    def _fallback_matching(self, baseline_nodules: List[NoduleInfo], followup_nodules: List[NoduleInfo],
                            matched_nodules: Dict[str, Tuple[NoduleInfo, NoduleInfo]], 
                            changes: Dict[str, Dict[str, float]]) -> None:
        """备用匹配策略（改进的简单匹配）
        
        Args:
            baseline_nodules: 基线结节列表
            followup_nodules: 随访结节列表
            matched_nodules: 存储匹配结果的字典，键为结节ID，值为(baseline, followup)元组
            changes: 存储变化指标的字典，键为结节ID，值为变化指标字典
            
        Returns:
            None
            
        Raises:
            ValueError: 如果输入参数类型无效
        """
        # 验证输入参数
        if not isinstance(baseline_nodules, list) or not isinstance(followup_nodules, list):
            logger.error("baseline_nodules and followup_nodules must be lists")
            raise ValueError("baseline_nodules and followup_nodules must be lists")
        
        if not isinstance(matched_nodules, dict) or not isinstance(changes, dict):
            logger.error("matched_nodules and changes must be dictionaries")
            raise ValueError("matched_nodules and changes must be dictionaries")
        
        logger.info(f"Using fallback matching for {len(baseline_nodules)} baseline and {len(followup_nodules)} followup nodules")
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
                
                if total_score > best_score and total_score > MATCH_SCORE_THRESHOLD:  # 使用常量替代魔法数字
                    best_score = total_score
                    best_match = f_nodule
            
            if best_match:
                nodule_id = b_nodule.nodule_id
                matched_nodules[nodule_id] = (b_nodule, best_match)
                changes[nodule_id] = self._calculate_nodule_changes(b_nodule, best_match)
                changes[nodule_id]["match_score"] = best_score
                changes[nodule_id]["fallback_matching"] = True  # 标记为备用匹配
                
                logger.debug(f"Fallback match found for nodule {nodule_id}: score={best_score:.3f}")
            else:
                logger.debug(f"No fallback match found for nodule {b_nodule.nodule_id}")
        
        logger.info(f"Fallback matching completed: {len(matched_nodules)} matches found")
    
    def _calculate_nodule_changes(self, baseline: NoduleInfo, followup: NoduleInfo) -> Dict[str, float]:
        """计算结节变化"""
        try:
            # 验证输入参数
            if baseline.volume_mm3 <= 0:
                logger.warning(f"Invalid baseline volume: {baseline.volume_mm3}")
                baseline_volume = 1e-6
            else:
                baseline_volume = baseline.volume_mm3
            
            if followup.volume_mm3 <= 0:
                logger.warning(f"Invalid followup volume: {followup.volume_mm3}")
                followup_volume = baseline_volume
            else:
                followup_volume = followup.volume_mm3
            
            volume_change = ((followup_volume - baseline_volume) / baseline_volume) * 100
            density_change = followup.mean_hu - baseline.mean_hu
            diameter_change = followup.diameter_mm - baseline.diameter_mm
            
            # 记录异常变化
            if abs(volume_change) > 1000:  # 体积变化超过1000%
                logger.warning(f"Extreme volume change detected: {volume_change:.1f}% for nodule {baseline.nodule_id}")
            
            if abs(density_change) > 1000:  # 密度变化超过1000HU
                logger.warning(f"Extreme density change detected: {density_change:.1f}HU for nodule {baseline.nodule_id}")
            
            return {
                "volume_change_percent": volume_change,
                "density_change_hu": density_change,
                "diameter_change_mm": diameter_change,
                "volume_ratio": followup_volume / baseline_volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating nodule changes for {baseline.nodule_id}: {e}")
            return {
                "volume_change_percent": 0.0,
                "density_change_hu": 0.0,
                "diameter_change_mm": 0.0,
                "volume_ratio": 1.0
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
        try:
            # 验证输出目录
            output_path = Path(output_file)
            output_dir = output_path.parent
            if not output_dir.exists():
                logger.info(f"Creating output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # 验证样本数据
            if not samples:
                logger.warning("No samples to save")
                return
            
            logger.info(f"Saving {len(samples)} samples to {output_file}")
            
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
                yaml.dump(dataset, f, allow_unicode=True, indent=2, sort_keys=False)

            logger.info(f"Dataset successfully saved to {output_file}")
            
        except PermissionError as e:
            logger.error(f"Permission denied when saving to {output_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving dataset to {output_file}: {e}")
            raise

    def generate_dataset(self, metadata_file: str, output_file: str, language: str = "zh"):
        """完整的数据集生成流程"""
        logger.info("Starting dataset generation...")
        
        try:
            # 验证输入参数
            if not metadata_file or not output_file:
                raise ValueError("metadata_file and output_file must be provided")
            
            if language not in ["zh", "en"]:
                logger.warning(f"Unsupported language '{language}', using 'zh'")
                language = "zh"
            
            logger.info(f"Configuration: volume_threshold={self.volume_threshold}, "
                       f"density_threshold={self.density_threshold}, "
                       f"min_nodule_volume={self.min_nodule_volume}, "
                       f"max_followup_days={self.max_followup_days}")
            
            logger.info("Loading LIDC metadata...")
            patient_nodules = self.load_lidc_data(metadata_file)
            
            if not patient_nodules:
                logger.warning("No patient data loaded from metadata")
                return
            
            logger.info("Finding longitudinal pairs...")
            longitudinal_pairs = self.find_longitudinal_pairs(patient_nodules)
            
            if not longitudinal_pairs:
                logger.warning("No longitudinal pairs found")
                return
            
            logger.info(f"Found {len(longitudinal_pairs)} longitudinal pairs")
            
            logger.info("Generating instruction-label pairs...")
            samples = self.generate_instruction_label_pairs(longitudinal_pairs, language)
            
            if not samples:
                logger.warning("No samples generated")
                return
            
            logger.info(f"Generated {len(samples)} samples")
            
            logger.info("Saving dataset...")
            self.save_dataset(samples, output_file)
            
            logger.info("Dataset generation completed successfully!")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid parameter: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during dataset generation: {e}")
            raise

        return samples

def create_lidc_longitudinal_dataset(data_root: str, metadata_file: str,
                                        output_file: str, config_file: Optional[str] = None,
                                        **kwargs) -> List[Dict[str, Any]]:
    """便捷函数：创建LIDC纵向数据集
    
    Args:
        data_root: 数据根目录
        metadata_file: 元数据文件路径
        output_file: 输出文件路径
        config_file: 配置文件路径（可选）
        **kwargs: 其他参数
        
    Returns:
        生成的样本列表
    """
    generator = LIDCLongitudinalDataGenerator(data_root, config_file=config_file, **kwargs)
    return generator.generate_dataset(metadata_file, output_file)

if __name__ == "__main__":
    # 示例用法 - 支持配置文件
    import argparse
    
    parser = argparse.ArgumentParser(description="LIDC Longitudinal Data Generator")
    parser.add_argument("--data_root", type=str, default="/path/to/lidc/data",
                       help="数据根目录")
    parser.add_argument("--metadata_file", type=str, default="/path/to/lidc_metadata.yaml",
                       help="元数据文件路径")
    parser.add_argument("--output_file", type=str, default="/path/to/lidc_longitudinal_dataset.yaml",
                       help="输出文件路径")
    parser.add_argument("--config_file", type=str, default="configs/longitudinal_lidc_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--volume_threshold", type=float, default=25.0,
                       help="体积变化阈值(%)")
    parser.add_argument("--density_threshold", type=float, default=50.0,
                       help="密度变化阈值(HU)")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"],
                       help="语言类型")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    config_file = args.config_file if Path(args.config_file).exists() else None
    if not config_file:
        logger.warning(f"Configuration file {args.config_file} not found, using default parameters")
    
    samples = create_lidc_longitudinal_dataset(
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        config_file=config_file,
        volume_threshold=args.volume_threshold,
        density_threshold=args.density_threshold,
        language=args.language
    )

    print(f"Generated {len(samples)} longitudinal samples")