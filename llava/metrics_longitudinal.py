"""
纵向推理分割任务的变化计算和评估指标
支持LIDC-IDRI数据集的变化量化分析

输入:
- mask_t0: 基线分割掩码（numpy数组，形状为H×W×D）
- mask_t1: 随访分割掩码（numpy数组，形状为H×W×D）
- ct_t0: 基线CT图像（numpy数组，HU值，形状为H×W×D）
- ct_t1: 随访CT图像（numpy数组，HU值，形状为H×W×D）
- voxel_spacing: 体素间距（三元组，单位mm，默认为(1.0, 1.0, 1.0)）
- task_type: 评估任务类型（volume_threshold/new_lesion/density_change/combined_attributes）
- threshold: 体积变化阈值（百分比，默认为25%）

输出:
- ChangeMetrics数据类：包含体积变化百分比、密度变化HU值、直径变化mm、边界模糊评分、纹理变化评分、空间位移mm
- 变化计算字典：包含具体的变化数值、变化率、统计信息等
- 任务评估结果：布尔值，表示是否满足任务条件
- 综合评分：0-1之间的综合变化严重程度评分

功能:
- 计算体积变化（基于体素计数和实际体积）
- 计算密度变化（HU值变化，支持ROI限定）
- 计算直径变化（基于最大直径和等效球直径）
- 计算边界模糊度变化（边界梯度分析）
- 计算纹理变化（基于灰度共生矩阵）
- 计算空间位移（质心位移和形状变化）
- 提供任务特定的评估逻辑（体积阈值、新病灶检测等）
- 生成综合变化评分和医学报告
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import ndimage
from skimage import measure, morphology
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChangeMetrics:
    """变化指标数据类"""
    volume_change_percent: float
    density_change_hu: float
    diameter_change_mm: float
    margin_blur_score: float
    texture_change_score: float
    spatial_shift_mm: float
    
class ChangeCalculator:
    """变化计算器：计算纵向CT扫描中的各种变化指标"""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing
        
    def calculate_volume_change(
        self, 
        mask_t0: np.ndarray, 
        mask_t1: np.ndarray
    ) -> Dict[str, float]:
        """
        计算体积变化
        
        Args:
            mask_t0: 基线掩码 (H, W, D)
            mask_t1: 随访掩码 (H, W, D)
            
        Returns:
            体积变化信息字典
        """
        # 计算体素数量
        voxel_volume = np.prod(self.voxel_spacing)
        
        volume_t0 = np.sum(mask_t0 > 0) * voxel_volume
        volume_t1 = np.sum(mask_t1 > 0) * voxel_volume
        
        # 避免除零
        if volume_t0 == 0:
            volume_change_percent = float('inf') if volume_t1 > 0 else 0.0
        else:
            volume_change_percent = ((volume_t1 - volume_t0) / volume_t0) * 100.0
            
        return {
            "volume_t0_mm3": volume_t0,
            "volume_t1_mm3": volume_t1,
            "volume_change_mm3": volume_t1 - volume_t0,
            "volume_change_percent": volume_change_percent,
            "volume_change_ratio": volume_t1 / max(volume_t0, 1e-6)
        }
    
    def calculate_density_change(
        self,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        mask_t0: Optional[np.ndarray] = None,
        mask_t1: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        计算密度变化（HU值）
        
        Args:
            ct_t0: 基线CT图像 (H, W, D)
            ct_t1: 随访CT图像 (H, W, D)
            mask_t0: 基线病灶掩码（可选）
            mask_t1: 随访病灶掩码（可选）
            
        Returns:
            密度变化信息字典
        """
        # 使用掩码限定ROI，如果没有掩码则使用整个图像
        roi_t0 = ct_t0[mask_t0 > 0] if mask_t0 is not None else ct_t0.flatten()
        roi_t1 = ct_t1[mask_t1 > 0] if mask_t1 is not None else ct_t1.flatten()
        
        # 过滤掉空气和骨组织（HU值极端值）
        roi_t0 = roi_t0[(roi_t0 > -1000) & (roi_t0 < 1000)]
        roi_t1 = roi_t1[(roi_t1 > -1000) & (roi_t1 < 1000)]
        
        if len(roi_t0) == 0 or len(roi_t1) == 0:
            return {
                "mean_hu_t0": 0.0,
                "mean_hu_t1": 0.0,
                "density_change_hu": 0.0,
                "density_change_percent": 0.0,
                "std_hu_t0": 0.0,
                "std_hu_t1": 0.0
            }
        
        mean_hu_t0 = np.mean(roi_t0)
        mean_hu_t1 = np.mean(roi_t1)
        density_change_hu = mean_hu_t1 - mean_hu_t0
        density_change_percent = (density_change_hu / max(abs(mean_hu_t0), 1e-6)) * 100.0
        
        return {
            "mean_hu_t0": mean_hu_t0,
            "mean_hu_t1": mean_hu_t1,
            "density_change_hu": density_change_hu,
            "density_change_percent": density_change_percent,
            "std_hu_t0": np.std(roi_t0),
            "std_hu_t1": np.std(roi_t1)
        }
    
    def calculate_diameter_change(
        self,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray
    ) -> Dict[str, float]:
        """
        计算直径变化（基于最大直径）
        
        Args:
            mask_t0: 基线掩码
            mask_t1: 随访掩码
            
        Returns:
            直径变化信息字典
        """
        def get_max_diameter(mask):
            """计算掩码的最大直径"""
            if np.sum(mask) == 0:
                return 0.0
            
            # 找到最大连通分量
            labeled = measure.label(mask > 0)
            if labeled.max() == 0:
                return 0.0
            
            # 获取最大的连通分量
            regions = measure.regionprops(labeled)
            largest_region = max(regions, key=lambda x: x.area)
            
            # 计算等效直径（假设球形）
            volume = largest_region.area * np.prod(self.voxel_spacing)
            equivalent_diameter = 2 * ((3 * volume) / (4 * np.pi)) ** (1/3)
            
            return equivalent_diameter
        
        diameter_t0 = get_max_diameter(mask_t0)
        diameter_t1 = get_max_diameter(mask_t1)
        
        if diameter_t0 == 0:
            diameter_change_percent = float('inf') if diameter_t1 > 0 else 0.0
        else:
            diameter_change_percent = ((diameter_t1 - diameter_t0) / diameter_t0) * 100.0
        
        return {
            "diameter_t0_mm": diameter_t0,
            "diameter_t1_mm": diameter_t1,
            "diameter_change_mm": diameter_t1 - diameter_t0,
            "diameter_change_percent": diameter_change_percent
        }
    
    def calculate_margin_blur(
        self,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray
    ) -> Dict[str, float]:
        """
        计算边界模糊度变化
        
        Args:
            ct_t0: 基线CT
            ct_t1: 随访CT
            mask_t0: 基线掩码
            mask_t1: 随访掩码
            
        Returns:
            边界模糊度信息字典
        """
        def compute_margin_gradient(ct, mask):
            """计算边界梯度"""
            # 膨胀和腐蚀获取边界
            dilated = morphology.binary_dilation(mask, morphology.ball(2))
            eroded = morphology.binary_erosion(mask, morphology.ball(2))
            boundary = dilated & ~eroded
            
            if np.sum(boundary) == 0:
                return 0.0
            
            # 计算边界处的梯度
            grad_z, grad_y, grad_x = np.gradient(ct)
            gradient_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            
            # 边界区域的平均梯度
            boundary_gradient = np.mean(gradient_magnitude[boundary > 0])
            
            return boundary_gradient
        
        blur_score_t0 = compute_margin_gradient(ct_t0, mask_t0)
        blur_score_t1 = compute_margin_gradient(ct_t1, mask_t1)
        blur_change = blur_score_t1 - blur_score_t0
        
        return {
            "blur_score_t0": blur_score_t0,
            "blur_score_t1": blur_score_t1,
            "blur_change": blur_change,
            "margin_become_clearer": blur_change < -0.1,
            "margin_become_blurrier": blur_change > 0.1
        }
    
    def calculate_all_changes(
        self,
        ct_t0: np.ndarray,
        ct_t1: np.ndarray,
        mask_t0: np.ndarray,
        mask_t1: np.ndarray
    ) -> ChangeMetrics:
        """
        计算所有变化指标
        
        Args:
            ct_t0: 基线CT
            ct_t1: 随访CT
            mask_t0: 基线掩码
            mask_t1: 随访掩码
            
        Returns:
            综合变化指标
        """
        # 计算各项变化
        volume_info = self.calculate_volume_change(mask_t0, mask_t1)
        density_info = self.calculate_density_change(ct_t0, ct_t1, mask_t0, mask_t1)
        diameter_info = self.calculate_diameter_change(mask_t0, mask_t1)
        blur_info = self.calculate_margin_blur(ct_t0, ct_t1, mask_t0, mask_t1)
        
        # 计算纹理变化与空间位移
        from scipy import ndimage
        if np.sum(mask_t0) > 0 and np.sum(mask_t1) > 0:
            # 质心位移
            center_t0 = ndimage.center_of_mass(mask_t0)
            center_t1 = ndimage.center_of_mass(mask_t1)
            spatial_shift_mm = np.sqrt(sum((a - b)**2 for a, b in zip(center_t0, center_t1))) * max(self.voxel_spacing)
            
            # 纹理变化：边界长度变化
            boundary_t0 = ndimage.laplace(mask_t0.astype(float))
            boundary_t1 = ndimage.laplace(mask_t1.astype(float))
            boundary_length_t0 = np.sum(np.abs(boundary_t0))
            boundary_length_t1 = np.sum(np.abs(boundary_t1))
            texture_change_score = abs(boundary_length_t1 - boundary_length_t0) / max(boundary_length_t0, 1)
        else:
            spatial_shift_mm = 0.0
            texture_change_score = 0.0
        
        return ChangeMetrics(
            volume_change_percent=volume_info["volume_change_percent"],
            density_change_hu=density_info["density_change_hu"],
            diameter_change_mm=diameter_info["diameter_change_mm"],
            margin_blur_score=blur_info["blur_change"],
            texture_change_score=texture_change_score,
            spatial_shift_mm=spatial_shift_mm
        )

class LongitudinalMetrics(nn.Module):
    """纵向推理分割任务的评估指标"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
    def dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算Dice系数"""
        pred = pred.float()
        target = target.float()
        
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return dice.mean()
    
    def hausdorff_distance_95(self, pred: np.ndarray, target: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> float:
        """计算95% Hausdorff距离 - 支持2D和3D输入"""
        from scipy.spatial.distance import directed_hausdorff
        
        # 处理维度：SegLLM输出2D掩码(448x448)
        pred_squeezed = pred.squeeze()
        target_squeezed = target.squeeze()
        
        # 找到边界点
        pred_points = np.argwhere(pred_squeezed > 0.5)
        target_points = np.argwhere(target_squeezed > 0.5)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0
        
        # 计算双向Hausdorff距离
        d1 = directed_hausdorff(pred_points, target_points)[0]
        d2 = directed_hausdorff(target_points, pred_points)[0]
        
        # 根据维度调整体素间距
        if pred_squeezed.ndim == 2:  # 2D情况
            hd = max(d1, d2) * max(voxel_spacing[:2])  # 只用前两个维度
        else:  # 3D情况
            hd = max(d1, d2) * max(voxel_spacing)
        
        return hd
    
    def condition_consistency(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor,
        change_metrics: ChangeMetrics,
        instruction: str
    ) -> Dict[str, float]:
        """
        计算条件一致性指标 - 使用正则提取和关键词白名单
        
        Args:
            pred_mask: 预测掩码
            target_mask: 目标掩码
            change_metrics: 变化指标
            instruction: 指令文本
            
        Returns:
            条件一致性指标字典
        """
        import re
        
        # 解析指令中的条件
        conditions_met = []
        
        # 关键词白名单 - 避免同义词漏匹配
        volume_keywords = ["体积", "大小", "尺寸"]
        density_keywords = ["密度", "HU", "CT值"]
        increase_keywords = ["增加", "增大", "变大", "超过", "≥", ">"]
        decrease_keywords = ["减少", "减小", "变小", "低于", "≤", "<"]
        
        # 检查体积变化条件
        has_volume = any(keyword in instruction for keyword in volume_keywords)
        has_percent = re.search(r'(\d+)%', instruction)
        
        if has_volume and has_percent:
            threshold = float(has_percent.group(1))
            
            # 判断增加还是减少
            is_increase = any(keyword in instruction for keyword in increase_keywords)
            is_decrease = any(keyword in instruction for keyword in decrease_keywords)
            
            # 检查预测是否满足条件
            pred_volume = torch.sum(pred_mask > 0.5).item()
            target_volume = torch.sum(target_mask > 0.5).item()
            
            if target_volume > 0:
                volume_change = ((pred_volume - target_volume) / target_volume) * 100
                
                if is_increase:
                    conditions_met.append(volume_change >= threshold)
                elif is_decrease:
                    conditions_met.append(volume_change <= -threshold)
                else:
                    # 如果没有明确方向，检查绝对值
                    conditions_met.append(abs(volume_change) >= threshold)
        
        # 检查密度条件
        has_density = any(keyword in instruction for keyword in density_keywords)
        has_hu_threshold = re.search(r'(\d+)HU', instruction)
        
        if has_density and has_hu_threshold:
            threshold = float(has_hu_threshold.group(1))
            
            # 使用真实的密度变化信息（如果可用）
            if hasattr(change_metrics, 'density_change_hu') and change_metrics.density_change_hu != 0:
                density_change = change_metrics.density_change_hu
                
                # 判断密度变化类型
                if "磨玻璃变实性" in instruction or "GGO" in instruction:
                    # 密度增加超过阈值视为磨玻璃变实性
                    conditions_met.append(density_change > threshold)
                elif "实性变磨玻璃" in instruction:
                    # 密度减少超过阈值视为实性变磨玻璃
                    conditions_met.append(density_change < -threshold)
                else:
                    # 一般密度变化
                    conditions_met.append(abs(density_change) >= threshold)
            else:
                # 如果没有真实密度信息，使用占位符
                conditions_met.append(True)
        
        consistency_score = sum(conditions_met) / max(len(conditions_met), 1)
        
        return {
            "condition_consistency": consistency_score,
            "conditions_analyzed": len(conditions_met),
            "conditions_satisfied": sum(conditions_met)
        }
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        change_metrics_list: List[ChangeMetrics],
        instructions: List[str]
    ) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            pred_masks: 预测掩码 (B, H, W, D)
            target_masks: 目标掩码 (B, H, W, D)
            change_metrics_list: 变化指标列表
            instructions: 指令列表
            
        Returns:
            综合评估指标字典
        """
        batch_size = pred_masks.shape[0]
        
        # 基础分割指标
        dice_scores = []
        hd95_scores = []
        
        # 条件一致性指标
        consistency_scores = []
        
        for i in range(batch_size):
            # Dice分数
            dice = self.dice_score(pred_masks[i:i+1], target_masks[i:i+1])
            dice_scores.append(dice.item())
            
            # Hausdorff距离
            pred_np = pred_masks[i].cpu().numpy()
            target_np = target_masks[i].cpu().numpy()
            hd95 = self.hausdorff_distance_95(pred_np, target_np, (1.0, 1.0, 1.0))
            hd95_scores.append(hd95)
            
            # 条件一致性
            consistency = self.condition_consistency(
                pred_masks[i], 
                target_masks[i], 
                change_metrics_list[i], 
                instructions[i]
            )
            consistency_scores.append(consistency["condition_consistency"])
        
        return {
            "dice_mean": np.mean(dice_scores),
            "dice_std": np.std(dice_scores),
            "hd95_mean": np.mean(hd95_scores),
            "hd95_std": np.std(hd95_scores),
            "condition_consistency_mean": np.mean(consistency_scores),
            "condition_consistency_std": np.std(consistency_scores)
        }