"""
SegLLM纵向对话模板
纵向推理对话模板和指令生成器，支持LIDC-IDRI纵向分割任务

输入:
- task_type: 任务类型（volume_threshold/new_lesion/density_change/combined_attributes）
- image_t0_path: 基线图像路径（T0时相）
- image_t1_path: 随访图像路径（T1时相）
- target_mask_path: 目标掩码路径
- change_info: 变化信息字典，包含volume_change、density_change、is_new等字段
- num_variations: 指令变体数量，用于数据增强

输出:
- instruction: 生成的指令文本，包含[IMAGE256]标记
- response: 响应文本，通常为[SEG]
- metadata: 元数据字典，包含target_mask_path、task_type等信息

功能:
- 管理四种纵向推理任务模板
- 生成符合医学术语的对话指令
- 支持指令变体生成以增加数据多样性
- 适配SegLLM框架的[IMAGE256]和[SEG]标记格式
- 提供任务权重调度和历史统计功能
"""

import copy
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class TaskTemplate:
    """任务模板"""
    task_type: str
    instruction_template: str
    response_template: str
    thresholds: Dict[str, float]
    description: str

class LongitudinalConversationTemplates:
    """纵向推理对话模板管理器"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.task_weights = {
            "volume_threshold": 0.3,
            "new_lesion": 0.25, 
            "density_change": 0.25,
            "combined_attributes": 0.2
        }
        self.task_history = {
            "volume_threshold": 0,
            "new_lesion": 0,
            "density_change": 0,
            "combined_attributes": 0
        }
    
    def _initialize_templates(self) -> Dict[str, TaskTemplate]:
        """初始化任务模板 - 适配SegLLM接口"""
        templates = {
            # T1: 体积阈值推理
            "volume_threshold": TaskTemplate(
                task_type="volume_threshold",
                instruction_template="[IMAGE256:{image_t0}|{image_t1}] 分割所有较上次体积{change_direction}{threshold}%的结节",
                response_template="[SEG]",
                thresholds={"increase": 25.0, "decrease": 20.0},
                description="基于体积变化的结节分割"
            ),
            
            # T2: 新发/消退病灶
            "new_lesion": TaskTemplate(
                task_type="new_lesion",
                instruction_template="[IMAGE256:{image_t0}|{image_t1}] 标出{lesion_type}的{lesion_subtype}结节",
                response_template="[SEG]",
                thresholds={},
                description="新出现或消失的病灶检测"
            ),
            
            # T3: 密度/形态变化
            "density_change": TaskTemplate(
                task_type="density_change",
                instruction_template="[IMAGE256:{image_t0}|{image_t1}] 圈出{change_type}的病灶",
                response_template="[SEG]",
                thresholds={"density_change": 50.0, "margin_blur": 0.3},
                description="密度和形态变化检测"
            ),
            
            # T4: 多属性组合
            "combined_attributes": TaskTemplate(
                task_type="combined_attributes",
                instruction_template="[IMAGE256:{image_t0}|{image_t1}] 分割{condition_desc}的结节",
                response_template="[SEG]",
                thresholds={"volume_change": 20.0, "density_threshold": 150.0},
                description="多条件组合筛选"
            )
        }
        return templates
    
    def generate_instruction(
        self,
        task_type: str,
        image_t0_path: str,
        image_t1_path: str,
        target_mask_path: str,
        change_info: Optional[Dict] = None
    ) -> Tuple[str, str, Dict]:
        """
        生成指令和响应 - 适配SegLLM接口
        
        Args:
            task_type: 任务类型
            image_t0_path: 基线图像路径
            image_t1_path: 随访图像路径  
            target_mask_path: 目标掩码路径
            change_info: 变化信息字典
            
        Returns:
            (instruction, response, metadata) 元组 - metadata包含掩码路径
        """
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        template = self.templates[task_type]
        
        # 根据任务类型填充模板
        if task_type == "volume_threshold":
            change_direction = "增加超过" if change_info.get("volume_change", 0) > 0 else "减少超过"
            threshold = abs(change_info.get("volume_change", 25))
            
            instruction = template.instruction_template.format(
                image_t0=image_t0_path,
                image_t1=image_t1_path,
                change_direction=change_direction,
                threshold=threshold
            )
            
        elif task_type == "new_lesion":
            lesion_type = "新出现" if change_info.get("is_new", True) else "消失"
            lesion_subtype = random.choice(["磨玻璃", "实性", "部分实性"])
            
            instruction = template.instruction_template.format(
                image_t0=image_t0_path,
                image_t1=image_t1_path,
                lesion_type=lesion_type,
                lesion_subtype=lesion_subtype
            )
            
        elif task_type == "density_change":
            density_change = change_info.get("density_change", 0)
            if density_change > 0:
                change_type = "从磨玻璃变实性"
            elif density_change < 0:
                change_type = "从实性变磨玻璃"
            else:
                change_type = "边界由清晰变模糊"
                
            instruction = template.instruction_template.format(
                image_t0=image_t0_path,
                image_t1=image_t1_path,
                change_type=change_type
            )
            
        elif task_type == "combined_attributes":
            conditions = []
            if change_info.get("volume_change", 0) >= 20:
                conditions.append(f"体积增加≥{change_info['volume_change']:.0f}%")
            if change_info.get("density_change", 0) >= 150:
                conditions.append(f"密度≥{change_info['density_change']:.0f}HU")
                
            condition_desc = "且".join(conditions) if conditions else "体积和密度均有显著变化"
            
            instruction = template.instruction_template.format(
                image_t0=image_t0_path,
                image_t1=image_t1_path,
                condition_desc=condition_desc
            )
        
        # 生成响应 - SegLLM格式
        response = template.response_template
        
        # 元数据包含掩码路径，供后续处理使用
        metadata = {
            "target_mask_path": target_mask_path,
            "image_t1_path": image_t1_path,
            "task_type": task_type
        }
        
        return instruction, response, metadata
    
    def generate_variations(
        self,
        base_instruction: str,
        base_response: str,
        base_metadata: Dict,
        num_variations: int = 3
    ) -> List[Tuple[str, str, Dict]]:
        """生成指令变体以增加多样性 - 适配SegLLM接口"""
        variations = []
        
        # 医学术语同义词白名单 - 避免误导性替换
        medical_synonyms = {
            "分割": ["标出", "圈出", "找出", "定位"],
            "结节": ["病灶", "阴影"],  # 移除"肿块"避免良性/恶性混淆
            "体积": ["大小", "尺寸"],
            "增加": ["增大", "变大"],
            "新出现": ["新发", "出现"],
            "磨玻璃": ["GGO", "毛玻璃"]
        }
        
        # 保护关键医学术语 - 这些词不应被替换
        protected_terms = ["实性", "良性", "恶性", "密度", "HU"]
        
        for i in range(num_variations):
            varied_instruction = base_instruction
            varied_response = base_response
            varied_metadata = base_metadata.copy()
            
            # 随机替换一些词语（保护关键医学术语）
            for word, syn_list in medical_synonyms.items():
                if word in varied_instruction and random.random() > 0.5:
                    replacement = random.choice(syn_list)
                    varied_instruction = varied_instruction.replace(word, replacement)
            
            # 轻微调整阈值表述
            if "超过" in varied_instruction:
                if random.random() > 0.7:
                    varied_instruction = varied_instruction.replace("超过", "大于")
            
            # 阈值随机扰动（±5%）提升边界值鲁棒性
            if "25%" in varied_instruction and random.random() > 0.5:
                varied_instruction = varied_instruction.replace("25%", f"{random.randint(23, 27)}%")
            elif "20%" in varied_instruction and random.random() > 0.5:
                varied_instruction = varied_instruction.replace("20%", f"{random.randint(18, 22)}%")
            
            variations.append((varied_instruction, varied_response, varied_metadata))
        
        return variations
    
    def create_few_shot_examples(
        self,
        task_type: str,
        num_examples: int = 2
    ) -> List[Dict]:
        """创建few-shot学习示例"""
        examples = []
        
        if task_type == "volume_threshold":
            examples = [
                {
                    "instruction": "[IMAGE256:example_t0.jpg] [IMAGE256:example_t1.jpg] 分割所有较上次体积增加超过30%的结节",
                    "response": "[MASK-DECODE:example_t1.jpg|INFERENCE|mask_30percent.nii]",
                    "explanation": "该结节体积从100mm³增加到135mm³，增长35%，超过30%阈值"
                },
                {
                    "instruction": "[IMAGE256:example2_t0.jpg] [IMAGE256:example2_t1.jpg] 找出体积减少超过25%的病灶",
                    "response": "[MASK-DECODE:example2_t1.jpg|INFERENCE|mask_decrease.nii]", 
                    "explanation": "病灶体积从200mm³减少到140mm³，减少30%，超过25%阈值"
                }
            ]
            
        elif task_type == "new_lesion":
            examples = [
                {
                    "instruction": "[IMAGE256:baseline.jpg] [IMAGE256:followup.jpg] 标出新出现的磨玻璃结节",
                    "response": "[MASK-DECODE:followup.jpg|INFERENCE|new_ggo_mask.nii]",
                    "explanation": "在随访扫描中发现一个新的磨玻璃密度结节，直径约8mm"
                }
            ]
            
        elif task_type == "density_change":
            examples = [
                {
                    "instruction": "[IMAGE256:scan1.jpg] [IMAGE256:scan2.jpg] 圈出从磨玻璃变实性的病灶",
                    "response": "[MASK-DECODE:scan2.jpg|INFERENCE|density_change_mask.nii]",
                    "explanation": "该结节密度从-600HU增加到-100HU，提示从磨玻璃向实性转变"
                }
            ]
            
        elif task_type == "combined_attributes":
            examples = [
                {
                    "instruction": "[IMAGE256:base.jpg] [IMAGE256:follow.jpg] 分割体积增加≥20%且密度≥150HU的结节",
                    "response": "[MASK-DECODE:follow.jpg|INFERENCE|combined_mask.nii]",
                    "explanation": "该结节体积增加25%且实性成分密度达到180HU，符合复合条件"
                }
            ]
        
        return examples[:num_examples]
    
    def validate_instruction_format(self, instruction: str, response: str, metadata: Optional[Dict] = None) -> bool:
        """验证指令格式是否正确 - 适配SegLLM接口"""
        # 检查必要的token
        required_tokens = ["[IMAGE256:", "[SEG]"]
        
        for token in required_tokens:
            if token not in instruction and token not in response:
                logger.warning(f"Missing required token: {token}")
                return False
        
        # 检查纵向图像格式 - 使用竖线拼接的双图像路径
        if "[IMAGE256:" in instruction:
            # 应该包含竖线拼接的双图像路径
            image_token_content = instruction[instruction.find("[IMAGE256:")+9:instruction.find("]", instruction.find("[IMAGE256:"))]
            if "|" not in image_token_content:
                logger.warning("Longitudinal task should have pipe-separated dual image paths in IMAGE256 token")
                return False
        
        # 检查响应格式
        if response != "[SEG]":
            logger.warning("Response should be exactly '[SEG]' for SegLLM compatibility")
            return False
            
        # 检查元数据
        if metadata is not None and "target_mask_path" not in metadata:
            logger.warning("Metadata should contain target_mask_path for SegLLM processing")
            return False
            
        return True

class LongitudinalTaskScheduler:
    """任务调度器，控制不同任务的训练比例"""
    
    def __init__(self, task_weights: Dict[str, float]):
        self.task_weights = task_weights
        self.task_history = {task: 0 for task in task_weights.keys()}
        
    def select_next_task(self) -> str:
        """选择下一个任务类型"""
        # 计算每个任务的采样概率
        total_weight = sum(self.task_weights.values())
        probabilities = [self.task_weights[task] / total_weight for task in self.task_weights.keys()]
        
        # 考虑历史采样次数，增加多样性
        tasks = list(self.task_weights.keys())
        adjusted_probs = []
        
        for i, task in enumerate(tasks):
            # 减少最近采样过的任务的概率
            history_factor = 1.0 / (1.0 + self.task_history[task] * 0.1)
            adjusted_prob = probabilities[i] * history_factor
            adjusted_probs.append(adjusted_prob)
        
        # 重新归一化概率
        total_adjusted = sum(adjusted_probs)
        normalized_probs = [p / total_adjusted for p in adjusted_probs]
        
        # 采样选择任务
        import numpy as np
        selected_task = np.random.choice(tasks, p=normalized_probs)
        
        # 更新历史记录
        self.task_history[selected_task] += 1
        
        return selected_task
    
    def get_task_statistics(self) -> Dict[str, int]:
        """获取任务采样统计"""
        return self.task_history.copy()
    
    def get_task_weights(self, difficulty_weighted: bool = True) -> Dict[str, float]:
        """获取任务权重，用于采样 - 支持难度权重"""
        # 基础权重
        base_weights = {
            "volume_threshold": 1.0,
            "new_lesion": 1.0,
            "density_change": 1.0,
            "combined_attributes": 1.0
        }
        
        if not difficulty_weighted:
            return base_weights
            
        # 难度权重 - 根据任务复杂度调整（课程学习）
        # combined_attributes 需要同时判断体积和密度，难度最高
        # new_lesion 需要检测新发病灶，难度较高
        # density_change 需要判断密度变化，中等难度
        # volume_threshold 只需要判断体积阈值，相对简单
        difficulty_weights = {
            "volume_threshold": 0.8,  # 简单任务，降低权重
            "density_change": 1.0,    # 中等任务，保持基础权重
            "new_lesion": 1.2,       # 困难任务，提高权重
            "combined_attributes": 1.5  # 最困难任务，最高权重
        }
        
        # 合并权重
        final_weights = {}
        for task in base_weights:
            final_weights[task] = base_weights[task] * difficulty_weights[task]
            
        # 归一化权重
        total_weight = sum(final_weights.values())
        normalized_weights = {task: weight/total_weight for task, weight in final_weights.items()}
        
        return normalized_weights
    
    def sample_task(self, difficulty_weighted: bool = True) -> str:
        """根据权重采样任务 - 支持难度权重"""
        weights = self.get_task_weights(difficulty_weighted)
        tasks = list(weights.keys())
        task_probs = list(weights.values())
        
        # 使用权重进行采样
        selected_task = random.choices(tasks, weights=task_probs, k=1)[0]
        
        # 更新任务历史
        self.task_history[selected_task] += 1
        
        return selected_task