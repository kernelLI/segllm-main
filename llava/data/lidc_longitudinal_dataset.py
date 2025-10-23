"""
SegLLM纵向推理分割数据集
LIDC-IDRI纵向推理分割数据集，支持双时相CT输入和变化推理任务

输入:
- data_path: 数据集根目录路径，包含longitudinal_pairs.json元数据文件和图像文件
- tokenizer: 文本分词器，用于处理对话文本
- data_args: 数据参数对象，包含image_size等配置信息
- image_processor: 图像处理器，用于图像预处理
- is_inference: 是否为推理模式，默认为False
- inference_conv: 推理模式下的对话配置，可选
- ct_window_center: CT窗位，默认为-600
- ct_window_width: CT窗宽，默认为1500

输出:
- 训练模式: 返回包含图像路径、掩码、对话和元数据的字典
- 推理模式: 返回处理后的对话数据

功能:
- 加载LIDC-IDRI纵向CT图像对（T0和T1时相）
- 支持NIfTI格式(.nii/.nii.gz)和普通图像格式
- 生成四种纵向推理任务：体积阈值、新发病灶、密度变化、多属性组合
- 创建多轮对话训练数据，支持[IMAGE256]和[SEG]特殊标记
- 处理CT窗宽窗位转换和切片选择
- 提供异常处理和虚拟数据回退机制
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import json
import os
from PIL import Image
import nibabel as nib
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import SimpleITK as sitk

logger = logging.getLogger(__name__)

@dataclass
class LongitudinalSample:
    """纵向样本数据结构"""
    patient_id: str
    study_t0: str  # 基线扫描
    study_t1: str  # 随访扫描
    image_t0_path: str
    image_t1_path: str
    mask_t0_path: str  # 现在支持list格式：可以是str或List[str]
    mask_t1_path: str  # 现在支持list格式：可以是str或List[str]
    nodules_info: List[Dict]  # 结节变化信息
    
class LIDCLongitudinalDataset(Dataset):
    """
    LIDC-IDRI纵向推理分割数据集
    支持双时相CT图像输入和变化推理
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_args,
        image_processor,
        is_inference: bool = False,
        inference_conv: Optional[Dict] = None,
        ct_window_center: int = -600,  # CT窗宽窗位配置
        ct_window_width: int = 1500
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_processor = image_processor
        self.is_inference = is_inference
        self.inference_conv = inference_conv
        self.ct_window_center = ct_window_center  # 配置化CT窗参数
        self.ct_window_width = ct_window_width
        
        # 从配置文件中获取图像尺寸，默认为256x256
        self.image_size = getattr(data_args, 'image_size', [256, 256, 64])[:2]
        self.default_height, self.default_width = self.image_size
        
        # 加载数据集元信息
        self.samples = self._load_samples()
        self.conversations = []
        
        if not is_inference:
            # 生成训练对话
            self.conversations = self._generate_conversations()
        else:
            # 推理模式使用提供的对话
            self.conversations = [inference_conv] if inference_conv else []
            
        logger.info(f"Loaded {len(self.samples)} longitudinal samples")
        logger.info(f"Generated {len(self.conversations)} conversations")
    
    def _load_samples(self) -> List[LongitudinalSample]:
        """加载纵向样本数据 - 支持四位医生标注"""
        samples = []
        
        # 加载数据集描述文件
        meta_file = os.path.join(self.data_path, "longitudinal_pairs.json")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
            
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
            
        for patient_data in meta_data:
            # 处理mask路径，支持单路径和多位医生路径列表
            def get_mask_paths(mask_field):
                mask_data = patient_data.get(mask_field, [])
                if isinstance(mask_data, str):
                    # 单路径格式
                    return os.path.join(self.data_path, mask_data)
                elif isinstance(mask_data, list):
                    # 多医生标注格式
                    return [os.path.join(self.data_path, path) for path in mask_data]
                else:
                    # 默认单路径
                    return os.path.join(self.data_path, str(mask_data))
            
            mask_t0_paths = get_mask_paths("mask_t0")
            mask_t1_paths = get_mask_paths("mask_t1")
            
            sample = LongitudinalSample(
                patient_id=patient_data["patient_id"],
                study_t0=patient_data["study_t0"],
                study_t1=patient_data["study_t1"],
                image_t0_path=os.path.join(self.data_path, patient_data["image_t0"]),
                image_t1_path=os.path.join(self.data_path, patient_data["image_t1"]),
                mask_t0_path=mask_t0_paths,
                mask_t1_path=mask_t1_paths,
                nodules_info=patient_data["nodules"]
            )
            samples.append(sample)
            
        return samples
    
    def _generate_conversations(self) -> List[Dict]:
        """生成训练对话数据"""
        conversations = []
        
        for sample in self.samples:
            # 为每个样本生成多种推理任务
            conv_templates = self._create_task_templates(sample)
            
            for template in conv_templates:
                conversation = {
                    "id": f"{sample.patient_id}_{template['task_id']}",
                    "image": [sample.image_t0_path, sample.image_t1_path],  # 双图像路径
                    "masks": {
                        "t0": sample.mask_t0_path,
                        "t1": sample.mask_t1_path,
                        "target": template["target_mask"]  # 目标掩码路径
                    },
                    "conversations": template["conversations"],
                    "metadata": {
                        "patient_id": sample.patient_id,
                        "study_t0": sample.study_t0,
                        "study_t1": sample.study_t1,
                        "task_type": template["task_type"],
                        "changes": template.get("changes", {})
                    }
                }
                conversations.append(conversation)
                
        return conversations
    
    def _create_task_templates(self, sample: LongitudinalSample) -> List[Dict]:
        """创建任务模板"""
        templates = []
        
        # T1: 体积阈值推理
        for nodule in sample.nodules_info:
            vol_change = nodule.get("volume_change_percent", 0)
            
            if vol_change >= 25:
                templates.append({
                    "task_id": f"t1_vol_increase_{nodule['id']}",
                    "task_type": "volume_threshold",
                    "target_mask": nodule["mask_t1_path"],
                    "changes": {"volume_change": vol_change},
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 分割所有较上次体积增加超过25%的结节"
                        },
                        {
                            "from": "gpt",
                            "value": f"[SEG]"
                        }
                    ]
                })
            
        # T2: 新发/消退病灶
        new_nodules = [n for n in sample.nodules_info if n.get("is_new", False)]
        for nodule in new_nodules:
            templates.append({
                "task_id": f"t2_new_{nodule['id']}",
                "task_type": "new_lesion",
                "target_mask": nodule["mask_t1_path"],
                "conversations": [
                    {
                        "from": "human", 
                        "value": f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 标出新出现的磨玻璃结节"
                    },
                    {
                        "from": "gpt",
                        "value": f"[SEG]"
                    }
                ]
            })
            
        # T3: 密度/形态变化
        for nodule in sample.nodules_info:
            density_change = nodule.get("density_change", 0)
            
            if abs(density_change) > 50:  # HU值变化超过50
                templates.append({
                    "task_id": f"t3_density_{nodule['id']}",
                    "task_type": "density_change",
                    "target_mask": nodule["mask_t1_path"],
                    "changes": {"density_change": density_change},
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 圈出从磨玻璃变实性的病灶"
                        },
                        {
                            "from": "gpt", 
                            "value": f"[SEG]"
                        }
                    ]
                })
                
        # T4: 多属性组合
        for nodule in sample.nodules_info:
            vol_change = nodule.get("volume_change_percent", 0)
            density_change = nodule.get("density_change", 0)
            
            if vol_change >= 20 and density_change >= 150:
                templates.append({
                    "task_id": f"t4_combined_{nodule['id']}",
                    "task_type": "combined_attributes",
                    "target_mask": nodule["mask_t1_path"],
                    "changes": {
                        "volume_change": vol_change,
                        "density_change": density_change
                    },
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 分割体积增加≥20%且密度≥150HU的结节"
                        },
                        {
                            "from": "gpt",
                            "value": f"[SEG]"
                        }
                    ]
                })
                
        return templates
    
    def _resample_ct_to_1mm(self, image_path: str) -> Optional[sitk.Image]:
        """将CT图像重采样到1mm层厚"""
        try:
            # 使用SimpleITK加载图像
            image_sitk = sitk.ReadImage(image_path)
            
            # 获取原始spacing和size
            original_spacing = image_sitk.GetSpacing()
            original_size = image_sitk.GetSize()
            
            # 如果z轴spacing已经是1mm，直接返回
            if abs(original_spacing[2] - 1.0) < 0.01:
                return image_sitk
                
            # 计算新的spacing和size
            new_spacing = list(original_spacing)
            new_spacing[2] = 1.0  # 设置z轴spacing为1mm
            
            # 根据新的spacing计算新的size，保持物理空间一致
            new_size = [int(round(osz * osp / nsp)) 
                       for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]
            
            # 创建重采样滤波器
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(image_sitk.GetDirection())
            resampler.SetOutputOrigin(image_sitk.GetOrigin())
            resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值
            
            # 执行重采样
            resampled_image = resampler.Execute(image_sitk)
            
            logger.info(f"Resampled {image_path}: spacing {original_spacing} -> {new_spacing}, size {original_size} -> {new_size}")
            return resampled_image
            
        except Exception as e:
            logger.error(f"Failed to resample CT image {image_path}: {str(e)}")
            return None
    
    def _load_ct_image(self, image_path: str) -> Image.Image:
        """加载CT图像 - 返回PIL.Image避免双重处理"""
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            # 首先进行层厚重采样到1mm
            resampled_sitk = self._resample_ct_to_1mm(image_path)
            
            # 配准失败回退处理
            if resampled_sitk is None:
                logger.error(f"CT resampling failed for {image_path}, returning empty image")
                return Image.new('RGB', (self.default_width, self.default_height), color='black')
            
            # 转换为numpy数组
            ct_data = sitk.GetArrayFromImage(resampled_sitk)
            
            # 选择关键切片（最大投影）
            if len(ct_data.shape) == 3:
                # 计算每个切片的前景像素数（二值化后求和）作为代理指标
                # 避免使用原始强度值，防止概率图影响选层
                binary_slices = (ct_data > -1000).astype(np.float32)  # 简单的二值化阈值
                slice_sums = np.sum(binary_slices, axis=(1, 2))  # sitk数组维度顺序为(z,y,x)
                slice_idx = np.argmax(slice_sums)
                ct_slice = ct_data[slice_idx, :, :]
            else:
                ct_slice = ct_data
                
            # HU值窗口化：使用配置的肺窗参数
            # 修复：window=-600, level=1500 应该对应区间 [-1500, 300] 而不是 [-1500, 600]
            # 正确的肺窗设置：window_center=-600, window_width=1500
            min_hu = self.ct_window_center - self.ct_window_width // 2  # -600 - 750 = -1350
            max_hu = self.ct_window_center + self.ct_window_width // 2  # -600 + 750 = 150
            ct_slice = np.clip((ct_slice - min_hu) / (max_hu - min_hu) * 255, 0, 255).astype(np.uint8)
            
            # 转换为RGB图像（三通道重复）
            ct_rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
            return Image.fromarray(ct_rgb)
            
        else:
            # 普通图像格式
            try:
                with Image.open(image_path) as img:
                    return img.convert('RGB')
            except Exception as e:
                logger.error(f"Error loading CT image {image_path}: {str(e)}")
                # 返回虚拟图像数据 - 使用配置的图像尺寸
                return Image.new('RGB', (self.default_width, self.default_height), color='black')
    
    def _select_random_doctor_mask(self, mask_paths) -> str:
        """从多位医生标注中随机选择一个"""
        if isinstance(mask_paths, list):
            # 如果是多位医生的标注列表，随机选择一个
            return np.random.choice(mask_paths)
        else:
            # 如果是单路径，直接返回
            return mask_paths
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """加载分割掩码"""
        # 先选择随机医生标注
        selected_path = self._select_random_doctor_mask(mask_path)
        
        if selected_path.endswith('.nii') or selected_path.endswith('.nii.gz'):
            nii_img = nib.load(selected_path)
            mask_data = nii_img.get_fdata()
            
            # 选择对应切片
            if len(mask_data.shape) == 3:
                slice_idx = mask_data.shape[2] // 2
                mask_slice = mask_data[:, :, slice_idx]
            else:
                mask_slice = mask_data
                
            mask = (mask_slice > 0).astype(np.uint8)
            
        else:
            try:
                with Image.open(selected_path) as mask_img:
                    mask = mask_img.convert('L')
                    mask = (np.array(mask) > 0).astype(np.uint8)
            except Exception as e:
                print(f"Error loading mask {selected_path}: {str(e)}")
                # 返回虚拟掩码数据 - 使用配置的图像尺寸
                mask = np.zeros((self.default_height, self.default_width), dtype=np.uint8)
            
        return torch.from_numpy(mask).long()
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """获取数据项 - 支持随机选择医生标注"""
        max_retries = 10
        original_idx = idx
        
        for retry in range(max_retries):
            try:
                conv = self.conversations[idx]
                
                if self.is_inference:
                    # 推理模式直接返回处理后的数据
                    return conv
                
                # 训练模式返回路径列表 - SegLLM期望的格式
                # 加载目标掩码（这里仍需加载，因为需要验证数据有效性）
                target_mask = self._load_mask(conv["masks"]["target"])
                
                # 构建输入数据 - 使用'image'键返回路径列表
                data_dict = {
                    "image": [conv["image"][0], conv["image"][1]],  # 路径列表格式
                    "masks": target_mask,
                    "conversations": conv["conversations"],
                    "metadata": conv["metadata"]
                }
                
                return data_dict
                
            except Exception as e:
                logger.error(f"Error loading sample {idx} (retry {retry + 1}/{max_retries}): {str(e)}")
                idx = (idx + 1) % len(self)
                if retry == max_retries - 1:
                    logger.error(f"Failed to load any valid sample after {max_retries} retries, returning None")
                    return None  # 返回None让collate_fn过滤
        
        # 不应该到达这里，但为了安全起见
        return None
    
    def collate_fn(self, batch):
        """
        自定义批次整理函数，处理不同大小的图像和掩码
        
        Args:
            batch: 批次数据列表
            
        Returns:
            整理后的批次数据
        """
        # 过滤掉None样本（加载失败的情况）
        valid_batch = [item for item in batch if item is not None]
        
        if len(valid_batch) == 0:
            # 如果所有样本都无效，返回空批次
            logger.warning("All samples in batch are None, returning empty batch")
            return {}
        
        # 提取各个字段
        batched_data = {}
        
        # 处理张量数据
        tensor_keys = ['input_ids', 'labels', 'attention_mask']
        for key in tensor_keys:
            if key in valid_batch[0] and valid_batch[0][key] is not None:
                try:
                    batched_data[key] = torch.stack([item[key] for item in valid_batch])
                except RuntimeError as e:
                    logger.warning(f"Failed to stack {key}: {e}, padding instead")
                    # 如果堆叠失败，进行填充
                    tensors = [item[key] for item in valid_batch]
                    max_len = max(t.shape[0] for t in tensors)
                    padded_tensors = []
                    for t in tensors:
                        if t.shape[0] < max_len:
                            padding = torch.zeros(max_len - t.shape[0], *t.shape[1:], dtype=t.dtype, device=t.device)
                            t = torch.cat([t, padding], dim=0)
                        padded_tensors.append(t)
                    batched_data[key] = torch.stack(padded_tensors)
                except Exception as e:
                    logger.error(f"Unexpected error stacking {key}: {e}")
                    # 返回空批次
                    return {}
        
        # 处理图像路径（保持为列表）
        if 'image' in valid_batch[0]:
            image_paths = []
            for item in valid_batch:
                if isinstance(item['image'], list):
                    image_paths.extend(item['image'])
                else:
                    image_paths.append(item['image'])
            batched_data['images'] = image_paths
        
        # 处理掩码路径（保持为列表）
        if 'masks' in valid_batch[0]:
            batched_data['masks'] = torch.stack([item['masks'] for item in valid_batch])
        
        # 处理目标掩码路径
        if 'target_mask' in valid_batch[0]:
            batched_data['target_mask'] = [item['target_mask'] for item in valid_batch]
        
        # 处理其他元数据
        metadata_keys = ['task_type', 'change_info', 'patient_id', 'conversations', 'metadata']
        for key in metadata_keys:
            if key in valid_batch[0]:
                batched_data[key] = [item[key] for item in valid_batch]
        
        # 添加批次大小信息
        batched_data['batch_size'] = len(valid_batch)
        
        return batched_data