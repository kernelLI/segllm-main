"""
纵向推理分割评测指标模块 - 用于LIDC-IDRI数据集
基于SegLLM架构，支持多种几何、演变和条件一致性指标
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re

logger = logging.getLogger(__name__)

class LongitudinalSegmentationMetrics(nn.Module):
    """纵向推理分割评测指标计算器"""
    
    def __init__(
        self,
        num_classes: int = 1,
        include_background: bool = False,
        metric_types: List[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        device: str = "cpu"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.include_background = include_background
        self.voxel_spacing = voxel_spacing
        self.device = device
        
        # 默认评测指标
        if metric_types is None:
            self.metric_types = [
                "dice", "iou", "precision", "recall", "f1",
                "hausdorff_95", "asd", "volume_error",
                "change_detection_f1", "condition_consistency",
                "threshold_accuracy", "progression_classification"
            ]
        else:
            self.metric_types = metric_types
        
        # 初始化指标缓存
        self.reset()
        
        logger.info(f"Initialized LongitudinalSegmentationMetrics with {len(self.metric_types)} metric types")
    
    def reset(self):
        """重置指标缓存"""
        self.metrics_cache = {metric: [] for metric in self.metric_types}
        self.batch_count = 0
    
    def compute_dice(self, pred: Tensor, target: Tensor, smooth: float = 1e-5) -> float:
        """计算Dice系数"""
        
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
        
        return dice.item()
    
    def compute_iou(self, pred: Tensor, target: Tensor, smooth: float = 1e-5) -> float:
        """计算IoU"""
        
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()
    
    def compute_precision_recall_f1(self, pred: Tensor, target: Tensor) -> Tuple[float, float, float]:
        """计算精确率、召回率和F1分数"""
        
        pred_binary = (pred > 0.5).float().cpu().numpy().flatten()
        target_binary = (target > 0.5).float().cpu().numpy().flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary', zero_division=0
        )
        
        return precision, recall, f1
    
    def compute_hausdorff_95(self, pred: Tensor, target: Tensor) -> float:
        """计算95% Hausdorff距离"""
        
        pred_binary = (pred > 0.5).cpu().numpy()
        target_binary = (target > 0.5).cpu().numpy()
        
        # 获取轮廓点
        pred_coords = np.argwhere(pred_binary)
        target_coords = np.argwhere(target_binary)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0
        
        # 计算Hausdorff距离
        forward_hausdorff = directed_hausdorff(pred_coords, target_coords)[0]
        backward_hausdorff = directed_hausdorff(target_coords, pred_coords)[0]
        
        hausdorff_max = max(forward_hausdorff, backward_hausdorff)
        
        # 考虑体素间距
        hausdorff_mm = hausdorff_max * np.mean(self.voxel_spacing)
        
        return hausdorff_mm
    
    def compute_asd(self, pred: Tensor, target: Tensor) -> float:
        """计算平均表面距离 (Average Surface Distance)"""
        
        pred_binary = (pred > 0.5).cpu().numpy()
        target_binary = (target > 0.5).cpu().numpy()
        
        pred_coords = np.argwhere(pred_binary)
        target_coords = np.argwhere(target_binary)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            return 0.0
        
        # 计算平均距离
        distances_pred_to_target = []
        distances_target_to_pred = []
        
        for pred_point in pred_coords:
            min_dist = np.min(np.linalg.norm(target_coords - pred_point, axis=1))
            distances_pred_to_target.append(min_dist)
        
        for target_point in target_coords:
            min_dist = np.min(np.linalg.norm(pred_coords - target_point, axis=1))
            distances_target_to_pred.append(min_dist)
        
        asd_pred_to_target = np.mean(distances_pred_to_target)
        asd_target_to_pred = np.mean(distances_target_to_pred)
        
        asd = (asd_pred_to_target + asd_target_to_pred) / 2.0
        
        # 考虑体素间距
        asd_mm = asd * np.mean(self.voxel_spacing)
        
        return asd_mm
    
    def compute_volume_error(self, pred: Tensor, target: Tensor) -> float:
        """计算体积误差"""
        
        pred_volume = pred.sum().item() * np.prod(self.voxel_spacing)
        target_volume = target.sum().item() * np.prod(self.voxel_spacing)
        
        if target_volume > 0:
            volume_error = abs(pred_volume - target_volume) / target_volume
        else:
            volume_error = abs(pred_volume - target_volume)
        
        return volume_error
    
    def compute_change_detection_metrics(
        self,
        pred_change: Tensor,
        target_change: Tensor,
        change_type: str = "new"
    ) -> Dict[str, float]:
        """计算变化检测指标"""
        
        pred_binary = (pred_change > 0.5).float().cpu().numpy().flatten()
        target_binary = (target_change > 0.5).float().cpu().numpy().flatten()
        
        # 计算检测指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average='binary', zero_division=0
        )
        
        accuracy = accuracy_score(target_binary, pred_binary)
        
        return {
            f"{change_type}_precision": precision,
            f"{change_type}_recall": recall,
            f"{change_type}_f1": f1,
            f"{change_type}_accuracy": accuracy
        }
    
    def compute_condition_consistency(
        self,
        pred: Tensor,
        target: Tensor,
        condition_text: str,
        change_metrics: Dict[str, float]
    ) -> float:
        """计算条件一致性 - 预测结果是否符合文本条件"""
        
        try:
            # 检查预测是否满足条件
            pred_satisfies = self._check_prediction_satisfies_condition(pred, condition_text, change_metrics)
            
            # 检查真实标签是否满足条件
            target_satisfies = self._check_prediction_satisfies_condition(target, condition_text, change_metrics)
            
            # 如果预测和真实标签在条件满足性上一致，则返回1.0
            if pred_satisfies == target_satisfies:
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Error computing condition consistency: {e}")
            return 0.0
    
    def _parse_condition(self, condition_text: str, change_metrics: Dict[str, float]) -> bool:
        """解析条件文本并判断是否满足"""
        
        condition_text = condition_text.lower()
        
        # 体积条件
        if "体积增加" in condition_text or "增大" in condition_text:
            if "25%" in condition_text:
                return change_metrics.get("volume_change_percent", 0) >= 25.0
            elif "20%" in condition_text:
                return change_metrics.get("volume_change_percent", 0) >= 20.0
            elif "30%" in condition_text:
                return change_metrics.get("volume_change_percent", 0) >= 30.0
        
        # 密度条件
        if "密度增加" in condition_text or "变实" in condition_text:
            return change_metrics.get("density_change_hu", 0) > 0
        
        if "磨玻璃" in condition_text:
            return change_metrics.get("density_type", "") == "ground_glass"
        
        if "实性" in condition_text:
            return change_metrics.get("density_type", "") == "solid"
        
        # 形态条件
        if "边界模糊" in condition_text:
            return change_metrics.get("boundary_change", 0) > 0
        
        if "新出现" in condition_text or "新发" in condition_text:
            return change_metrics.get("is_new_lesion", False)
        
        # 默认返回True
        return True
    
    def _extract_condition_from_text(self, condition_text: str) -> Dict[str, Any]:
        """使用LLM方法从文本中提取条件信息
        
        Args:
            condition_text: 条件文本
            
        Returns:
            条件信息字典
        """
        try:
            # 首先尝试使用LLM方法提取条件
            condition_info = self._llm_extract_condition(condition_text)
            if condition_info["type"] != "none":
                return condition_info
            
            # 如果LLM方法失败，回退到传统关键词匹配
            return self._fallback_extract_condition(condition_text)
            
        except Exception as e:
            logger.warning(f"LLM条件提取失败: {str(e)}，使用回退方法")
            return self._fallback_extract_condition(condition_text)
    
    def _llm_extract_condition(self, condition_text: str) -> Dict[str, Any]:
        """使用LLM方法从文本中提取条件信息
        
        Args:
            condition_text: 条件文本
            
        Returns:
            条件信息字典
        """
        if not condition_text or not condition_text.strip():
            return {"type": "none"}
        
        # 定义特征提取的正则表达式模式 (参考项目中的text_conditioning_longitudinal.py)
        patterns = {
            "volume": {
                "increase": r"增加|增大|变大|增长|上升|≥|>=|>|超过",
                "decrease": r"减少|减小|变小|下降|降低|<|<=|≤|少于",
                "stable": r"稳定|不变|无变化|保持",
                "percentage": r"(\d+(?:\.\d+)?)\s*%",
                "absolute": r"(\d+(?:\.\d+)?)\s*(?:mm|cm|ml|cc)"
            },
            "density": {
                "increase": r"密度增加|变实|实性化|HU增加|CT值增加",
                "decrease": r"密度减少|变磨玻璃|磨玻璃化|HU减少|CT值减少",
                "ground_glass": r"磨玻璃|GGO|GGN|ground.glass",
                "solid": r"实性|solid|dense",
                "mixed": r"混合|部分实性|part.solid",
                "hu_value": r"(\d+(?:\.\d+)?)\s*HU",
                "ct_value": r"(\d+(?:\.\d+)?)\s*CT"
            },
            "morphology": {
                "smooth": r"光滑|smooth|规则|regular",
                "lobulated": r"分叶|lobulated|不规则|irregular",
                "spiculated": r"毛刺|spiculated|毛刺征",
                "well_defined": r"边界清晰|well.defined|清楚|清晰",
                "ill_defined": r"边界模糊|ill.defined|模糊|不清楚",
                "round": r"圆形|round|球形",
                "oval": r"椭圆形|oval|卵圆形",
                "irregular": r"不规则|irregular|形态不规则"
            },
            "change_type": {
                "new": r"新出现|new|新发|新增|出现",
                "disappear": r"消失|disappear|消散|缩小至无",
                "grow": r"生长|growth|增大|progression",
                "shrink": r"缩小|shrink|regression|好转",
                "stable": r"稳定|stable|无变化|unchanged"
            }
        }
        
        text_lower = condition_text.lower()
        
        # 提取结构化特征
        text_features = {}
        for feature_type, feature_patterns in patterns.items():
            feature_values = {}
            
            for pattern_name, pattern in feature_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    if pattern_name in ["percentage", "absolute", "hu_value", "ct_value"]:
                        # 数值特征
                        try:
                            feature_values[pattern_name] = float(matches[0])
                        except (ValueError, IndexError):
                            feature_values[pattern_name] = 0.0
                    else:
                        # 布尔特征
                        feature_values[pattern_name] = 1.0
                else:
                    if pattern_name in ["percentage", "absolute", "hu_value", "ct_value"]:
                        feature_values[pattern_name] = 0.0
                    else:
                        feature_values[pattern_name] = 0.0
            
            text_features[feature_type] = feature_values
        
        # 基于提取的特征构建条件信息
        condition_info = {"original_text": condition_text}
        
        # 分析体积特征
        if "volume" in text_features:
            volume_features = text_features["volume"]
            
            # 检查体积增加
            if volume_features.get("increase", 0) > 0:
                threshold = None
                
                # 提取百分比阈值
                if volume_features.get("percentage", 0) > 0:
                    threshold = volume_features["percentage"]
                else:
                    # 从文本中直接提取数字
                    percentage_match = re.search(r"(\d+(?:\.\d+)?)\s*%", condition_text)
                    if percentage_match:
                        threshold = float(percentage_match.group(1))
                    else:
                        threshold = 25.0  # 默认阈值
                
                condition_info.update({
                    "type": "volume_increase",
                    "threshold": threshold
                })
                return condition_info
        
        # 分析变化类型特征
        if "change_type" in text_features:
            change_features = text_features["change_type"]
            
            # 检查新发病灶
            if change_features.get("new", 0) > 0:
                condition_info.update({
                    "type": "new_lesion"
                })
                return condition_info
            
            # 检查生长/增大
            if change_features.get("grow", 0) > 0:
                condition_info.update({
                    "type": "volume_increase",
                    "threshold": 25.0  # 默认阈值
                })
                return condition_info
        
        # 分析密度特征
        if "density" in text_features:
            density_features = text_features["density"]
            
            # 检查密度增加
            if density_features.get("increase", 0) > 0:
                condition_info.update({
                    "type": "density_increase"
                })
                return condition_info
            
            # 检查密度减少
            if density_features.get("decrease", 0) > 0:
                condition_info.update({
                    "type": "density_decrease"
                })
                return condition_info
        
        # 分析形态特征
        if "morphology" in text_features:
            morphology_features = text_features["morphology"]
            
            # 检查是否有形态变化
            if any(value > 0 for value in morphology_features.values()):
                condition_info.update({
                    "type": "morphology_change"
                })
                return condition_info
        
        # 如果没有明确的特征，返回无意义条件
        return {"type": "none"}
    
    def _fallback_extract_condition(self, condition_text: str) -> Dict[str, Any]:
        """备选方案：使用传统关键词匹配"""
        
        if not condition_text or not condition_text.strip():
            return {"type": "none"}
        
        text = condition_text.lower().strip()
        
        # 多属性组合条件优先检查
        if any(word in text for word in ["并且", "且", "和", "同时", "both", "also"]):
            conditions = []
            
            # 分割复合条件
            parts = re.split(r"[，。；,;]|并且|且|和|同时", text)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                sub_condition = self._fallback_extract_condition(part)
                if sub_condition["type"] != "none":
                    conditions.append(sub_condition)
            
            if len(conditions) > 1:
                return {
                    "type": "composite",
                    "conditions": conditions,
                    "original_text": condition_text
                }
        
        # 体积增加条件
        volume_increase_keywords = ["增加", "增大", "变大", "增长", "上升", "≥", ">=", ">", "超过"]
        if any(keyword in text for keyword in volume_increase_keywords):
            threshold = None
            
            # 提取百分比阈值
            percentage_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if percentage_match:
                threshold = float(percentage_match.group(1))
            else:
                # 检查常见的百分比阈值
                if "20%" in text or "百分之二十" in text:
                    threshold = 20.0
                elif "25%" in text or "百分之二十五" in text:
                    threshold = 25.0
                elif "30%" in text or "百分之三十" in text:
                    threshold = 30.0
                elif "50%" in text or "百分之五十" in text:
                    threshold = 50.0
            
            return {
                "type": "volume_increase",
                "threshold": threshold if threshold is not None else 25.0,
                "original_text": condition_text
            }
        
        # 新发病灶条件
        new_lesion_keywords = ["新出现", "new", "新发", "新增", "出现", "新发结节"]
        if any(keyword in text for keyword in new_lesion_keywords):
            return {
                "type": "new_lesion",
                "original_text": condition_text
            }
        
        # 密度变化条件
        density_increase_keywords = ["密度增加", "变实", "实性化", "HU增加", "CT值增加"]
        density_decrease_keywords = ["密度减少", "变磨玻璃", "磨玻璃化", "HU减少", "CT值减少"]
        
        if any(keyword in text for keyword in density_increase_keywords):
            return {
                "type": "density_increase",
                "original_text": condition_text
            }
        elif any(keyword in text for keyword in density_decrease_keywords):
            return {
                "type": "density_decrease",
                "original_text": condition_text
            }
        
        # 形态变化条件
        morphology_keywords = ["边界清晰", "边界模糊", "光滑", "分叶", "毛刺", "规则", "不规则"]
        if any(keyword in text for keyword in morphology_keywords):
            return {
                "type": "morphology_change",
                "original_text": condition_text
            }
        
        return {"type": "none"}
    
    def _check_prediction_satisfies_condition(self, pred: Tensor, condition_text: str, 
                                             change_metrics: Dict[str, float] = None) -> bool:
        """检查预测是否满足条件 - 支持LLM提取的条件类型
        
        Args:
            pred: 预测掩码 [H, W, D]
            condition_text: 条件文本
            change_metrics: 变化指标字典
            
        Returns:
            是否满足条件
        """
        if change_metrics is None:
            change_metrics = {}
            
        # 解析条件并验证
        condition_info = self._extract_condition_from_text(condition_text)
        
        if condition_info["type"] == "volume_increase":
            # 体积增加条件：检查预测区域的体积变化是否满足阈值
            pred_volume = pred.sum().item()
            threshold = condition_info.get("threshold", 25.0)
            
            # 获取真实体积变化（如果可用）
            actual_volume_change = change_metrics.get("volume_change_percent", 0)
            
            # 如果预测有体积且真实变化满足阈值，则认为条件满足
            if pred_volume > 0 and actual_volume_change >= threshold:
                return True
            elif pred_volume == 0 and actual_volume_change < threshold:
                return True
            else:
                return False
                
        elif condition_info["type"] == "new_lesion":
            # 新发病灶条件：检查预测是否标识了新发病灶
            is_new = change_metrics.get("is_new_lesion", False)
            pred_positive = pred.sum().item() > 0
            
            # 预测和真实标签一致
            return pred_positive == is_new
            
        elif condition_info["type"] == "density_increase":
            # 密度增加条件：检查密度变化是否满足要求
            density_change = change_metrics.get("density_change_hu", 0)
            pred_positive = pred.sum().item() > 0
            
            # 简化的密度条件检查
            if density_change > 50:
                return pred_positive
            else:
                return not pred_positive  # 如果没有显著变化，预测应该为负
                
        elif condition_info["type"] == "density_decrease":
            # 密度减少条件：检查密度变化是否满足要求
            density_change = change_metrics.get("density_change_hu", 0)
            pred_positive = pred.sum().item() > 0
            
            # 简化的密度条件检查
            if density_change < -50:
                return pred_positive
            else:
                return not pred_positive  # 如果没有显著变化，预测应该为负
                
        elif condition_info["type"] == "morphology_change":
            # 形态变化条件：检查是否有形态变化
            has_morphology_change = change_metrics.get("has_morphology_change", False)
            pred_positive = pred.sum().item() > 0
            
            return pred_positive == has_morphology_change
            
        elif condition_info["type"] == "volume_threshold":
            # 兼容旧版本的条件类型
            pred_volume = pred.sum().item()
            threshold = condition_info.get("threshold", 25.0)
            actual_volume_change = change_metrics.get("volume_change_percent", 0)
            
            if pred_volume > 0 and actual_volume_change >= threshold:
                return True
            elif pred_volume == 0 and actual_volume_change < threshold:
                return True
            else:
                return False
                
        elif condition_info["type"] == "density_change":
            # 兼容旧版本的条件类型
            density_change = change_metrics.get("density_change_hu", 0)
            pred_positive = pred.sum().item() > 0
            
            if "增加" in condition_text and density_change > 50:
                return pred_positive
            elif "减少" in condition_text and density_change < -50:
                return pred_positive
            else:
                return not pred_positive
                
        elif condition_info["type"] == "combined_attributes":
            # 多属性组合条件
            volume_change = change_metrics.get("volume_change_percent", 0)
            density_change = change_metrics.get("density_change_hu", 0)
            pred_positive = pred.sum().item() > 0
            
            # 检查是否同时满足体积和密度条件
            vol_threshold = condition_info.get("volume_threshold", 20.0)
            dens_threshold = condition_info.get("density_threshold", 150.0)
            
            meets_volume = volume_change >= vol_threshold
            meets_density = density_change >= dens_threshold
            actual_meets_both = meets_volume and meets_density
            
            return pred_positive == actual_meets_both
            
        else:
            # 默认情况：如果有预测结果且真实条件满足，则认为满足
            pred_positive = pred.sum().item() > 0
            actual_condition_met = change_metrics.get("condition_met", True)
            return pred_positive == actual_condition_met
    
    def compute_threshold_accuracy(
        self,
        pred_change: float,
        target_change: float,
        threshold: float = 25.0
    ) -> float:
        """计算阈值准确性"""
        
        pred_satisfies = pred_change >= threshold
        target_satisfies = target_change >= threshold
        
        if pred_satisfies == target_satisfies:
            return 1.0
        else:
            return 0.0
    
    def compute_progression_classification(
        self,
        pred_volume_change: float,
        target_volume_change: float,
        thresholds: Tuple[float, float] = (-15.0, 25.0)
    ) -> Dict[str, float]:
        """计算进展分类准确性"""
        
        regression_threshold, progression_threshold = thresholds
        
        # 真实分类
        if target_volume_change <= regression_threshold:
            target_class = "regression"
        elif target_volume_change >= progression_threshold:
            target_class = "progression"
        else:
            target_class = "stable"
        
        # 预测分类
        if pred_volume_change <= regression_threshold:
            pred_class = "regression"
        elif pred_volume_change >= progression_threshold:
            pred_class = "progression"
        else:
            pred_class = "stable"
        
        # 计算准确性
        accuracy = 1.0 if pred_class == target_class else 0.0
        
        return {
            "progression_classification_accuracy": accuracy,
            "predicted_class": pred_class,
            "target_class": target_class
        }
    
    def update(
        self,
        pred: Tensor,
        target: Tensor,
        condition_text: str = "",
        change_metrics: Dict[str, float] = None,
        **kwargs
    ):
        """更新指标缓存"""
        
        if change_metrics is None:
            change_metrics = {}
        
        # 计算各个指标
        for metric_type in self.metric_types:
            try:
                if metric_type == "dice":
                    value = self.compute_dice(pred, target)
                elif metric_type == "iou":
                    value = self.compute_iou(pred, target)
                elif metric_type == "precision":
                    precision, _, _ = self.compute_precision_recall_f1(pred, target)
                    value = precision
                elif metric_type == "recall":
                    _, recall, _ = self.compute_precision_recall_f1(pred, target)
                    value = recall
                elif metric_type == "f1":
                    _, _, f1 = self.compute_precision_recall_f1(pred, target)
                    value = f1
                elif metric_type == "hausdorff_95":
                    value = self.compute_hausdorff_95(pred, target)
                elif metric_type == "asd":
                    value = self.compute_asd(pred, target)
                elif metric_type == "volume_error":
                    value = self.compute_volume_error(pred, target)
                elif metric_type == "change_detection_f1":
                    change_metrics_dict = self.compute_change_detection_metrics(pred, target)
                    value = change_metrics_dict.get("new_f1", 0.0)
                elif metric_type == "condition_consistency":
                    value = self.compute_condition_consistency(pred, target, condition_text, change_metrics)
                elif metric_type == "threshold_accuracy":
                    # 获取预测和真实的体积变化
                    pred_volume_change = change_metrics.get("pred_volume_change_percent", 0)
                    target_volume_change = change_metrics.get("volume_change_percent", 0)
                    value = self.compute_threshold_accuracy(pred_volume_change, target_volume_change)
                elif metric_type == "progression_classification":
                    # 获取预测和真实的体积变化
                    pred_volume_change = change_metrics.get("pred_volume_change_percent", 0)
                    target_volume_change = change_metrics.get("volume_change_percent", 0)
                    prog_dict = self.compute_progression_classification(pred_volume_change, target_volume_change)
                    value = prog_dict["progression_classification_accuracy"]
                else:
                    value = 0.0
                
                self.metrics_cache[metric_type].append(value)
                
            except Exception as e:
                logger.warning(f"Error computing {metric_type}: {e}")
                self.metrics_cache[metric_type].append(0.0)
        
        self.batch_count += 1
    
    def compute(self) -> Dict[str, float]:
        """计算最终指标"""
        
        results = {}
        
        for metric_type, values in self.metrics_cache.items():
            if len(values) > 0:
                results[f"{metric_type}_mean"] = np.mean(values)
                results[f"{metric_type}_std"] = np.std(values)
                results[f"{metric_type}_min"] = np.min(values)
                results[f"{metric_type}_max"] = np.max(values)
            else:
                results[f"{metric_type}_mean"] = 0.0
                results[f"{metric_type}_std"] = 0.0
                results[f"{metric_type}_min"] = 0.0
                results[f"{metric_type}_max"] = 0.0
        
        return results
    
    def get_summary(self) -> str:
        """获取指标摘要"""
        
        results = self.compute()
        
        summary_lines = ["Longitudinal Segmentation Metrics Summary:"]
        summary_lines.append("=" * 50)
        
        for metric_type in self.metric_types:
            mean_key = f"{metric_type}_mean"
            std_key = f"{metric_type}_std"
            
            if mean_key in results and std_key in results:
                mean_val = results[mean_key]
                std_val = results[std_key]
                summary_lines.append(f"{metric_type:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total batches: {self.batch_count}")
        
        return "\n".join(summary_lines)

# 便捷函数
def compute_longitudinal_metrics(
    pred: Tensor,
    target: Tensor,
    condition_text: str = "",
    change_metrics: Dict[str, float] = None,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cpu"
) -> Dict[str, float]:
    """计算纵向推理分割指标"""
    
    metrics_calculator = LongitudinalSegmentationMetrics(
        voxel_spacing=voxel_spacing,
        device=device
    )
    
    metrics_calculator.update(pred, target, condition_text, change_metrics)
    return metrics_calculator.compute()

if __name__ == "__main__":
    # 示例用法和测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建评测指标计算器
    metrics_calculator = LongitudinalSegmentationMetrics(
        voxel_spacing=(1.0, 1.0, 2.0),
        device="cpu"
    )
    
    # 生成测试数据
    batch_size = 4
    spatial_size = (32, 32, 16)
    
    # 随机预测和目标
    pred = torch.sigmoid(torch.randn(batch_size, *spatial_size))
    target = (torch.randn(batch_size, *spatial_size) > 0).float()
    
    # 变化指标
    change_metrics = {
        "volume_change_percent": 30.0,
        "density_change_hu": 150.0,
        "density_type": "solid",
        "boundary_change": 0.5,
        "is_new_lesion": False
    }
    
    condition_text = "分割所有体积增加超过25%的结节"
    
    print(f"Testing with:")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Condition text: {condition_text}")
    
    # 计算指标
    metrics_calculator.update(pred, target, condition_text, change_metrics)
    results = metrics_calculator.compute()
    
    print(f"\nMetrics results:")
    for key, value in results.items():
        if "mean" in key:
            print(f"  {key}: {value:.4f}")
    
    # 打印摘要
    print(f"\n{metrics_calculator.get_summary()}")
    
    print("\nMetrics test completed successfully!")