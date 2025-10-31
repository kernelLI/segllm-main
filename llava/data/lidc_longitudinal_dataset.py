"""
SegLLM纵向推理分割数据集
LIDC-IDRI纵向推理分割数据集，支持双时相CT输入和变化推理任务

输入:
- data_path: 数据集根目录路径，包含longitudinal_pairs.yaml元数据文件和图像文件
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
import yaml
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
        # 使用已配置的CT窗参数
        self.ct_windows = data_args.ct_windows
        
        # 从配置文件中获取图像尺寸，默认为256x256
        # 注意：使用2D尺寸，忽略Z维度
        if hasattr(data_args, 'image_size') and isinstance(data_args.image_size, (list, tuple)):
            self.image_size = data_args.image_size[:2]  # 只取前两个维度
        else:
            self.image_size = [256, 256]  # 默认2D尺寸
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
        
        # 只使用YAML格式文件
        meta_file = None
        for ext in ['.yaml', '.yml']:
            candidate_file = os.path.join(self.data_path, f"longitudinal_pairs{ext}")
            if os.path.exists(candidate_file):
                meta_file = candidate_file
                break
        
        if not meta_file:
            raise FileNotFoundError(f"未找到YAML格式的配对文件: longitudinal_pairs.yaml/yml in {self.data_path}")
            
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta_data = yaml.safe_load(f)
            
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
                # 根据密度变化类型生成不同的指令
                if density_change > 50:  # 密度增加
                    if density_change > 150:  # 显著增加，可能是磨玻璃变实性
                        instruction = f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 圈出从磨玻璃变实性的病灶"
                    else:  # 中等程度增加
                        instruction = f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 标出密度增加超过{density_change:.0f}HU的结节"
                else:  # 密度减少
                    if density_change < -150:  # 显著减少，可能是实性变磨玻璃
                        instruction = f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 找出从实性变磨玻璃的病灶"
                    else:  # 中等程度减少
                        instruction = f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 标出密度减少超过{abs(density_change):.0f}HU的结节"
                
                templates.append({
                    "task_id": f"t3_density_{nodule['id']}",
                    "task_type": "density_change",
                    "target_mask": nodule["mask_t1_path"],
                    "changes": {"density_change": density_change},
                    "conversations": [
                        {
                            "from": "human",
                            "value": instruction
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
            
            # 定义多种组合条件
            conditions = []
            
            # 条件1: 体积增加 + 密度增加
            if vol_change >= 20 and density_change >= 150:
                conditions.append({
                    "desc": "体积增加≥20%且密度≥150HU",
                    "priority": 1
                })
            
            # 条件2: 体积显著增加 + 密度中等增加
            if vol_change >= 50 and density_change >= 100:
                conditions.append({
                    "desc": "体积增加≥50%且密度增加≥100HU",
                    "priority": 2
                })
            
            # 条件3: 体积减少 + 密度减少（可能为治疗响应）
            if vol_change <= -30 and density_change <= -100:
                conditions.append({
                    "desc": f"体积减少≥30%且密度减少≥{abs(density_change):.0f}HU",
                    "priority": 3
                })
            
            # 条件4: 体积稳定但密度显著变化
            if -10 <= vol_change <= 10 and abs(density_change) >= 200:
                change_type = "增加" if density_change > 0 else "减少"
                conditions.append({
                    "desc": f"体积稳定但密度{change_type}≥{abs(density_change):.0f}HU",
                    "priority": 4
                })
            
            # 选择最高优先级的条件生成任务
            if conditions:
                best_condition = min(conditions, key=lambda x: x["priority"])
                
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
                            "value": f"[IMAGE256:{sample.image_t0_path}] [IMAGE256:{sample.image_t1_path}] 分割{best_condition['desc']}的结节"
                        },
                        {
                            "from": "gpt",
                            "value": f"[SEG]"
                        }
                    ]
                })
                
        return templates
    
    def _load_ct_image(self, image_path: str):
        """加载CT图像，返回(PIL.Image, raw_hu_slice)元组"""
        if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            nii_img = nib.load(image_path)
            ct_data = nii_img.get_fdata()

            # 选关键切片
            if len(ct_data.shape) == 3:
                slice_sums = np.sum(ct_data, axis=(0, 1))
                slice_idx = np.argmax(slice_sums)
                hu_slice = ct_data[:, :, slice_idx].astype(np.float32)
            else:
                hu_slice = ct_data.astype(np.float32)

            # 使用已配置的CT窗参数
            windows = self.ct_windows

            # 强制3窗
            ch_list = []
            for w in windows[:3]:
                wc, ww = w['center'], w['width']
                min_hu = wc - ww // 2
                max_hu = wc + ww // 2
                ch = np.clip((hu_slice - min_hu) / (max_hu - min_hu) * 255, 0, 255).astype(np.uint8)
                ch_list.append(ch)
            while len(ch_list) < 3:
                ch_list.append(ch_list[-1])
            rgb = np.stack(ch_list, axis=-1)
            return Image.fromarray(rgb), hu_slice
        else:
            # 普通图像格式兜底，也返回虚拟HU全0
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    return img, np.zeros((img.height, img.width), dtype=np.float32)
            except Exception as e:
                logger.error(f"Error loading CT image {image_path}: {str(e)}")
                # 返回虚拟图像数据 - 使用配置的图像尺寸
                return Image.new('RGB', (self.default_width, self.default_height), color='black'), \
                       np.zeros((self.default_height, self.default_width), dtype=np.float32)
    
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
        """获取样本数据"""
        max_retries = 10
        current_idx = idx
        
        for attempt in range(max_retries):
            try:
                # 获取配对数据
                conv = self.conversations[current_idx]
                
                if self.is_inference:
                    # 推理模式直接返回处理后的数据
                    return conv
                
                # 加载目标掩码
                target_mask = self._load_mask(conv["masks"]["target"])
                
                # 检查数据有效性
                if target_mask is None:
                    logging.warning(f"Invalid mask for sample {current_idx}, trying next sample")
                    current_idx = (current_idx + 1) % len(self.conversations)
                    continue
                
                # 加载t0/t1并生成变化热图与HU缓存
                img_t0, hu_t0 = self._load_ct_image(conv["image"][0])
                img_t1, hu_t1 = self._load_ct_image(conv["image"][1])
                
                # 检查图像有效性
                if img_t0 is None or img_t1 is None or hu_t0 is None or hu_t1 is None:
                    logging.warning(f"Invalid CT images for sample {current_idx}, trying next sample")
                    current_idx = (current_idx + 1) % len(self.conversations)
                    continue
                
                delta_hu = hu_t1 - hu_t0
                heatmap = np.clip((delta_hu + 200) / 400 * 255, 0, 255).astype(np.uint8)

                # 构建输入数据
                data_dict = {
                    "image": [conv["image"][0], conv["image"][1]],  # 仍给路径供外部二次加载
                    "masks": target_mask,
                    "conversations": conv["conversations"],
                    "metadata": conv["metadata"],
                    "hu_t0": hu_t0,          # 原始HU
                    "hu_t1": hu_t1,          # 原始HU
                    "change_heatmap": heatmap # 变化热图
                }
                
                return data_dict
                
            except Exception as e:
                logging.warning(f"Error loading sample {current_idx}: {e}")
                # 尝试下一个样本
                current_idx = (current_idx + 1) % len(self.conversations)
                
                # 如果是最后一个尝试，返回None让collate_fn处理
                if attempt == max_retries - 1:
                    return None
        
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