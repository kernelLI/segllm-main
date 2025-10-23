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

logger = logging.getLogger(__name__)

@dataclass
class LongitudinalSample:
    """纵向样本数据结构"""
    patient_id: str
    study_t0: str  # 基线扫描
    study_t1: str  # 随访扫描
    image_t0_path: str
    image_t1_path: str
    mask_t0_path: str  
    mask_t1_path: str
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
        """加载纵向样本数据"""
        samples = []
        
        # 加载数据集描述文件
        meta_file = os.path.join(self.data_path, "longitudinal_pairs.json")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
            
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)
            
        for patient_data in meta_data:
            sample = LongitudinalSample(
                patient_id=patient_data["patient_id"],
                study_t0=patient_data["study_t0"],
                study_t1=patient_data["study_t1"],
                image_t0_path=os.path.join(self.data_path, patient_data["image_t0"]),
                image_t1_path=os.path.join(self.data_path, patient_data["image_t1"]),
                mask_t0_path=os.path.join(self.data_path, patient_data["mask_t0"]),
                mask_t1_path=os.path.join(self.data_path, patient_data["mask_t1"]),
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
    
    def _load_ct_image(self, image_path: str) -> Image.Image:
        """加载CT图像 - 返回PIL.Image避免双重处理"""
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            # 加载NIfTI格式的CT图像
            nii_img = nib.load(image_path)
            ct_data = nii_img.get_fdata()
            
            # 选择关键切片（最大投影）
            if len(ct_data.shape) == 3:
                # 计算每个切片的总强度作为代理指标
                slice_sums = np.sum(ct_data, axis=(0, 1))
                slice_idx = np.argmax(slice_sums)
                ct_slice = ct_data[:, :, slice_idx]
            else:
                ct_slice = ct_data
                
            # HU值窗口化：使用配置的肺窗参数
            # 范围: [window_center - window_width/2, window_center + window_width/2]
            min_hu = self.ct_window_center - self.ct_window_width // 2
            max_hu = self.ct_window_center + self.ct_window_width // 2
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
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """加载分割掩码"""
        if mask_path.endswith('.nii') or mask_path.endswith('.nii.gz'):
            nii_img = nib.load(mask_path)
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
                with Image.open(mask_path) as mask_img:
                    mask = mask_img.convert('L')
                    mask = (np.array(mask) > 0).astype(np.uint8)
            except Exception as e:
                print(f"Error loading mask {mask_path}: {str(e)}")
                # 返回虚拟掩码数据 - 使用配置的图像尺寸
                mask = np.zeros((self.default_height, self.default_width), dtype=np.uint8)
            
        return torch.from_numpy(mask).long()
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        """获取数据项 - 返回路径列表格式"""
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
        """批处理函数 - 返回路径列表格式"""
        if self.is_inference:
            return batch[0] if batch else {}
            
        # 过滤掉None样本
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return {"image": [], "masks": [], "conversations": [], "metadata": []}
            
        # 提取路径列表 - SegLLM期望的格式
        image_paths = []
        for item in valid_batch:
            # 从__getitem__返回的"image"键获取路径列表
            if "image" in item and isinstance(item["image"], list):
                image_paths.extend(item["image"])
        
        return {
            "image": image_paths,  # SegLLM期望的路径列表格式
            "masks": torch.stack([item["masks"] for item in valid_batch]) if valid_batch else torch.zeros(0),
            "conversations": [item["conversations"] for item in valid_batch],
            "metadata": [item["metadata"] for item in valid_batch]
        }