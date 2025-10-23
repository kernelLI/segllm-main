"""
SegLLM结节配准与匹配模块
基于IOU、空间距离和形态特征自动匹配同一病人的结节

功能:
- 计算IOU和空间距离
- 提取形态特征（长径、短径、球形度）
- 综合匹配分数计算
- 自动生成结节ID映射表
"""

import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from scipy import ndimage
from skimage import measure, morphology
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class NoduleRegistration:
    """结节配准与匹配器"""
    
    def __init__(self, iou_weight: float = 0.4, distance_weight: float = 0.3, shape_weight: float = 0.3):
        """
        初始化配准器
        
        Args:
            iou_weight: IOU权重
            distance_weight: 空间距离权重
            shape_weight: 形态特征权重
        """
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.shape_weight = shape_weight
    
    def compute_shape_descriptor(self, mask: np.ndarray, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
        """
        计算形态描述符
        
        Args:
            mask: 二值掩码
            voxel_spacing: 体素间距 (mm)
            
        Returns:
            形态特征字典
        """
        if np.sum(mask) == 0:
            return {"long_axis": 0.0, "short_axis": 0.0, "sphericity": 0.0, "volume": 0.0}
        
        # 获取连通分量
        labeled = measure.label(mask > 0)
        if labeled.max() == 0:
            return {"long_axis": 0.0, "short_axis": 0.0, "sphericity": 0.0, "volume": 0.0}
        
        # 获取最大连通分量
        regions = measure.regionprops(labeled, spacing=voxel_spacing)
        largest_region = max(regions, key=lambda x: x.area)
        
        # 计算体积
        volume = largest_region.area * np.prod(voxel_spacing)
        
        # 计算长径和短径（基于边界框）
        bbox = largest_region.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
        bbox_size = np.array([bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]])
        bbox_size_mm = bbox_size * np.array(voxel_spacing)
        
        long_axis = np.max(bbox_size_mm)
        short_axis = np.min(bbox_size_mm)
        
        # 计算球形度 = (6 * V * sqrt(pi)) / A^(3/2)
        # 其中V是体积，A是表面积
        surface_area = largest_region.perimeter * np.mean(voxel_spacing)  # 近似
        if surface_area > 0:
            sphericity = (6 * volume * np.sqrt(np.pi)) / (surface_area ** (3/2))
            sphericity = min(sphericity, 1.0)  # 限制在[0,1]
        else:
            sphericity = 0.0
        
        return {
            "long_axis": float(long_axis),
            "short_axis": float(short_axis),
            "sphericity": float(sphericity),
            "volume": float(volume)
        }
    
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个掩码的IOU"""
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        return intersection / max(union, 1e-6)
    
    def compute_center_distance(self, mask1: np.ndarray, mask2: np.ndarray, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """计算两个掩码中心点的欧氏距离"""
        center1 = ndimage.center_of_mass(mask1)
        center2 = ndimage.center_of_mass(mask2)
        
        if len(center1) != len(center2):
            return float('inf')
        
        # 转换为物理坐标
        center1_phys = np.array(center1) * np.array(voxel_spacing)
        center2_phys = np.array(center2) * np.array(voxel_spacing)
        
        return euclidean(center1_phys, center2_phys)
    
    def compute_shape_similarity(self, shape1: Dict[str, float], shape2: Dict[str, float]) -> float:
        """计算形态相似度"""
        # 体积相似度
        vol1, vol2 = shape1["volume"], shape2["volume"]
        if vol1 == 0 and vol2 == 0:
            volume_sim = 1.0
        elif vol1 == 0 or vol2 == 0:
            volume_sim = 0.0
        else:
            volume_sim = min(vol1, vol2) / max(vol1, vol2)
        
        # 长径相似度
        long1, long2 = shape1["long_axis"], shape2["long_axis"]
        if long1 == 0 and long2 == 0:
            long_sim = 1.0
        elif long1 == 0 or long2 == 0:
            long_sim = 0.0
        else:
            long_sim = min(long1, long2) / max(long1, long2)
        
        # 球形度相似度
        sphere1, sphere2 = shape1["sphericity"], shape2["sphericity"]
        sphere_sim = 1.0 - abs(sphere1 - sphere2)
        
        # 综合形态相似度
        return (volume_sim + long_sim + sphere_sim) / 3.0
    
    def compute_matching_score(self, mask_t0: np.ndarray, mask_t1: np.ndarray, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        计算结节匹配分数
        
        Args:
            mask_t0: 基线结节掩码
            mask_t1: 随访结节掩码
            voxel_spacing: 体素间距
            
        Returns:
            匹配分数 [0,1]
        """
        # 计算IOU
        iou = self.compute_iou(mask_t0, mask_t1)
        
        # 计算中心距离（转换为相似度，距离越小相似度越高）
        center_dist = self.compute_center_distance(mask_t0, mask_t1, voxel_spacing)
        distance_sim = np.exp(-center_dist / 50.0)  # 50mm作为衰减常数
        
        # 计算形态相似度
        shape_t0 = self.compute_shape_descriptor(mask_t0, voxel_spacing)
        shape_t1 = self.compute_shape_descriptor(mask_t1, voxel_spacing)
        shape_sim = self.compute_shape_similarity(shape_t0, shape_t1)
        
        # 综合分数
        total_score = (self.iou_weight * iou + 
                      self.distance_weight * distance_sim + 
                      self.shape_weight * shape_sim)
        
        return total_score
    
    def register_nodules(self, masks_t0: List[np.ndarray], masks_t1: List[np.ndarray], 
                        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, any]:
        """
        配准两个时相的结节
        
        Args:
            masks_t0: 基线结节掩码列表
            masks_t1: 随访结节掩码列表
            voxel_spacing: 体素间距
            
        Returns:
            配准结果字典
        """
        n_t0 = len(masks_t0)
        n_t1 = len(masks_t1)
        
        # 计算所有配对的匹配分数
        scores = np.zeros((n_t0, n_t1))
        for i in range(n_t0):
            for j in range(n_t1):
                scores[i, j] = self.compute_matching_score(masks_t0[i], masks_t1[j], voxel_spacing)
        
        # 使用匈牙利算法找到最佳匹配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-scores)  # 注意：需要最大化分数
        
        # 构建映射表
        mapping = []
        new_nodules = []  # 新出现的结节
        disappeared_nodules = []  # 消失的结节
        
        for j in range(n_t1):
            if j in col_ind:
                # 找到匹配的基线结节
                i = row_ind[col_ind == j][0]
                score = scores[i, j]
                
                if score > 0.3:  # 匹配阈值
                    mapping.append({
                        "t0_id": i,
                        "t1_id": j,
                        "score": score,
                        "iou": self.compute_iou(masks_t0[i], masks_t1[j]),
                        "center_distance_mm": self.compute_center_distance(masks_t0[i], masks_t1[j], voxel_spacing),
                        "shape_similarity": self.compute_shape_similarity(
                            self.compute_shape_descriptor(masks_t0[i], voxel_spacing),
                            self.compute_shape_descriptor(masks_t1[j], voxel_spacing)
                        )
                    })
                else:
                    # 分数太低，认为是新结节
                    new_nodules.append(j)
            else:
                # 没有匹配的随访结节，认为是新出现的
                new_nodules.append(j)
        
        # 找到消失的结节
        for i in range(n_t0):
            if i not in row_ind:
                disappeared_nodules.append(i)
        
        return {
            "mapping": mapping,
            "new_nodules": new_nodules,
            "disappeared_nodules": disappeared_nodules,
            "total_t0": n_t0,
            "total_t1": n_t1,
            "matched": len(mapping),
            "matching_matrix": scores.tolist()
        }
    
    def save_mapping_to_json(self, mapping_result: Dict, output_path: str):
        """保存映射结果到JSON文件"""
        # 转换为可序列化格式
        serializable_result = {
            "mapping": mapping_result["mapping"],
            "new_nodules": mapping_result["new_nodules"],
            "disappeared_nodules": mapping_result["disappeared_nodules"],
            "total_t0": mapping_result["total_t0"],
            "total_t1": mapping_result["total_t1"],
            "matched": mapping_result["matched"]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved nodule mapping to {output_path}")


def compute_nodule_registration(masks_t0: List[np.ndarray], masks_t1: List[np.ndarray], 
                               voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                               output_path: Optional[str] = None) -> Dict:
    """
    便捷的结节配准函数
    
    Args:
        masks_t0: 基线结节掩码列表
        masks_t1: 随访结节掩码列表
        voxel_spacing: 体素间距
        output_path: 可选的输出JSON路径
        
    Returns:
        配准结果
    """
    registrar = NoduleRegistration()
    result = registrar.register_nodules(masks_t0, masks_t1, voxel_spacing)
    
    if output_path:
        registrar.save_mapping_to_json(result, output_path)
    
    return result


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    test_mask1 = np.zeros((50, 50, 20))
    test_mask1[10:30, 15:35, 8:12] = 1
    
    test_mask2 = np.zeros((50, 50, 20))
    test_mask2[12:32, 17:37, 8:12] = 1  # 稍微移动
    
    test_mask3 = np.zeros((50, 50, 20))
    test_mask3[5:15, 5:15, 5:10] = 1  # 不同的结节
    
    # 测试配准
    registrar = NoduleRegistration()
    result = registrar.register_nodules([test_mask1, test_mask3], [test_mask2])
    
    print("Registration result:")
    print(f"Matched: {result['matched']}")
    print(f"New nodules: {result['new_nodules']}")
    print(f"Disappeared nodules: {result['disappeared_nodules']}")
    print(f"Mapping: {result['mapping']}")