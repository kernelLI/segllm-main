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
    
    def _check_prediction_satisfies_condition(self, pred: Tensor, condition_text: str, 
                                             change_metrics: Dict[str, float] = None) -> bool:
        """检查预测是否满足条件
        
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
        
        if condition_info["type"] == "volume_threshold":
            # 体积阈值条件：检查预测区域的体积变化是否满足阈值
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
            
        elif condition_info["type"] == "density_change":
            # 密度变化条件：检查密度变化是否满足要求
            density_change = change_metrics.get("density_change_hu", 0)
            pred_positive = pred.sum().item() > 0
            
            # 简化的密度条件检查
            if "增加" in condition_text and density_change > 50:
                return pred_positive
            elif "减少" in condition_text and density_change < -50:
                return pred_positive
            else:
                return not pred_positive  # 如果没有显著变化，预测应该为负
                
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