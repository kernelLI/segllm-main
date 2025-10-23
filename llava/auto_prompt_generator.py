"""
SegLLM自动指令生成器
根据变化指标自动生成正向、扰动、反向三条文字指令

功能:
- 根据ΔV、ΔHU、边界模糊度自动生成指令
- 支持正向、±5%扰动、反向三条指令
- 支持多种任务类型模板
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class AutoPromptGenerator:
    """自动指令生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.templates = {
            "volume_threshold": {
                "positive": [
                    "分割体积{change_direction}{threshold}%的结节",
                    "标出体积{change_direction}{threshold}%的病灶",
                    "圈出体积变化{change_direction}{threshold}%的肺结节"
                ],
                "perturbed": [
                    "分割体积{change_direction}{perturbed_threshold}%的结节",
                    "标出体积{change_direction}{perturbed_threshold}%的病灶",
                    "圈出体积变化{change_direction}{perturbed_threshold}%的肺结节"
                ],
                "reverse": [
                    "分割体积{reverse_change_direction}{reverse_threshold}%的结节",
                    "标出体积{reverse_change_direction}{reverse_threshold}%的病灶",
                    "圈出体积变化{reverse_change_direction}{reverse_threshold}%的肺结节"
                ]
            },
            "density_change": {
                "positive": [
                    "分割密度变化为{density_change}HU的病灶",
                    "标出密度变化{density_change}HU的结节",
                    "圈出密度变化为{density_change}HU的肺结节"
                ],
                "perturbed": [
                    "分割密度变化为{perturbed_density_change}HU的病灶",
                    "标出密度变化{perturbed_density_change}HU的结节",
                    "圈出密度变化为{perturbed_density_change}HU的肺结节"
                ],
                "reverse": [
                    "分割密度变化为{reverse_density_change}HU的病灶",
                    "标出密度变化{reverse_density_change}HU的结节",
                    "圈出密度变化为{reverse_density_change}HU的肺结节"
                ]
            },
            "margin_change": {
                "positive": [
                    "分割边界{blur_direction}的结节",
                    "标出边界{blur_direction}的病灶",
                    "圈出边界{blur_direction}的肺结节"
                ],
                "perturbed": [
                    "分割边界{perturbed_blur_direction}的结节",
                    "标出边界{perturbed_blur_direction}的病灶",
                    "圈出边界{perturbed_blur_direction}的肺结节"
                ],
                "reverse": [
                    "分割边界{reverse_blur_direction}的结节",
                    "标出边界{reverse_blur_direction}的病灶",
                    "圈出边界{reverse_blur_direction}的肺结节"
                ]
            },
            "new_lesion": {
                "positive": [
                    "标出新出现的{lesion_type}结节",
                    "分割新出现的{lesion_type}病灶",
                    "圈出新出现的{lesion_type}肺结节"
                ],
                "perturbed": [
                    "标出新出现的{perturbed_lesion_type}结节",
                    "分割新出现的{perturbed_lesion_type}病灶",
                    "圈出新出现的{perturbed_lesion_type}肺结节"
                ],
                "reverse": [
                    "标出消失的{lesion_type}结节",
                    "分割消失的{lesion_type}病灶",
                    "圈出消失的{lesion_type}肺结节"
                ]
            }
        }
    
    def generate_three_prompts(self, task_type: str, change_metrics: Dict) -> Tuple[str, str, str]:
        """
        生成三条指令：正向、扰动、反向
        
        Args:
            task_type: 任务类型
            change_metrics: 变化指标字典，包含volume_change、density_change、blur_score等
            
        Returns:
            (正向指令, 扰动指令, 反向指令)
        """
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        template = self.templates[task_type]
        
        # 生成正向指令
        positive_prompt = self._generate_single_prompt(template["positive"], change_metrics, perturbation=0)
        
        # 生成扰动指令（±5%）
        perturbed_prompt = self._generate_single_prompt(template["perturbed"], change_metrics, perturbation=0.05)
        
        # 生成反向指令
        reverse_prompt = self._generate_reverse_prompt(template["reverse"], change_metrics)
        
        return positive_prompt, perturbed_prompt, reverse_prompt
    
    def _generate_single_prompt(self, template_list: List[str], metrics: Dict, perturbation: float = 0) -> str:
        """生成单条指令"""
        template = np.random.choice(template_list)
        
        # 根据任务类型填充模板
        if "volume_change" in metrics:
            volume_change = metrics["volume_change"]
            
            # 应用扰动
            if perturbation != 0:
                volume_change = volume_change * (1 + np.random.uniform(-perturbation, perturbation))
                volume_change = round(volume_change, 1)
            
            change_direction = "增加超过" if volume_change > 0 else "减少超过"
            threshold = abs(volume_change)
            
            template = template.format(
                change_direction=change_direction,
                threshold=threshold
            )
        
        elif "density_change" in metrics:
            density_change = metrics["density_change"]
            
            # 应用扰动
            if perturbation != 0:
                density_change = density_change * (1 + np.random.uniform(-perturbation, perturbation))
                density_change = round(density_change, 1)
            
            template = template.format(density_change=density_change)
        
        elif "blur_score" in metrics:
            blur_change = metrics.get("blur_change", 0)
            blur_threshold = 0.1
            
            # 应用扰动
            if perturbation != 0:
                blur_threshold = blur_threshold * (1 + np.random.uniform(-perturbation, perturbation))
            
            if blur_change > blur_threshold:
                blur_direction = "变模糊"
            elif blur_change < -blur_threshold:
                blur_direction = "变清晰"
            else:
                blur_direction = "无明显变化"
            
            template = template.format(blur_direction=blur_direction)
        
        elif "is_new" in metrics:
            is_new = metrics["is_new"]
            lesion_type = metrics.get("lesion_type", "磨玻璃")
            
            if perturbation != 0 and np.random.random() < 0.3:
                # 30%概率改变病灶类型
                lesion_types = ["磨玻璃", "实性", "混合", "钙化"]
                lesion_type = np.random.choice([t for t in lesion_types if t != lesion_type])
            
            template = template.format(lesion_type=lesion_type)
        
        return template
    
    def _generate_reverse_prompt(self, template_list: List[str], metrics: Dict) -> str:
        """生成反向指令"""
        template = np.random.choice(template_list)
        reverse_metrics = {}
        
        if "volume_change" in metrics:
            volume_change = metrics["volume_change"]
            # 反向：增加变减少，减少变增加
            reverse_volume_change = -volume_change
            reverse_metrics["reverse_change_direction"] = "增加超过" if reverse_volume_change > 0 else "减少超过"
            reverse_metrics["reverse_threshold"] = abs(reverse_volume_change)
            
            template = template.format(**reverse_metrics)
        
        elif "density_change" in metrics:
            density_change = metrics["density_change"]
            # 反向：密度变化取反
            reverse_density_change = -density_change
            reverse_metrics["reverse_density_change"] = reverse_density_change
            
            template = template.format(**reverse_metrics)
        
        elif "blur_score" in metrics:
            blur_change = metrics.get("blur_change", 0)
            # 反向：模糊度变化取反
            reverse_blur_change = -blur_change
            
            if reverse_blur_change > 0.1:
                reverse_metrics["reverse_blur_direction"] = "变模糊"
            elif reverse_blur_change < -0.1:
                reverse_metrics["reverse_blur_direction"] = "变清晰"
            else:
                reverse_metrics["reverse_blur_direction"] = "无明显变化"
            
            template = template.format(**reverse_metrics)
        
        elif "is_new" in metrics:
            # 反向：新出现变消失
            lesion_type = metrics.get("lesion_type", "磨玻璃")
            reverse_metrics["lesion_type"] = lesion_type
            
            template = template.format(**reverse_metrics)
        
        return template
    
    def generate_auto_prompts(self, change_metrics: Dict) -> Dict[str, Tuple[str, str, str]]:
        """
        自动生成所有任务类型的三条指令
        
        Args:
            change_metrics: 变化指标字典
            
        Returns:
            任务类型到三条指令的映射
        """
        results = {}
        
        # 根据变化指标确定任务类型
        if "volume_change" in change_metrics:
            volume_change = abs(change_metrics["volume_change"])
            if volume_change >= 25:  # 25%阈值
                results["volume_threshold"] = self.generate_three_prompts("volume_threshold", change_metrics)
        
        if "density_change" in change_metrics:
            density_change = abs(change_metrics["density_change"])
            if density_change >= 50:  # 50HU阈值
                results["density_change"] = self.generate_three_prompts("density_change", change_metrics)
        
        if "blur_change" in change_metrics:
            blur_change = abs(change_metrics["blur_change"])
            if blur_change >= 0.1:  # 0.1阈值
                results["margin_change"] = self.generate_three_prompts("margin_change", change_metrics)
        
        if "is_new" in change_metrics:
            results["new_lesion"] = self.generate_three_prompts("new_lesion", change_metrics)
        
        return results
    
    def save_prompts_to_json(self, prompts: Dict, output_path: str):
        """保存指令到JSON文件"""
        serializable_prompts = {}
        
        for task_type, (positive, perturbed, reverse) in prompts.items():
            serializable_prompts[task_type] = {
                "positive": positive,
                "perturbed": perturbed,
                "reverse": reverse
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved auto-generated prompts to {output_path}")


def generate_auto_prompts(change_metrics: Dict, output_path: Optional[str] = None) -> Dict:
    """
    便捷的自动指令生成函数
    
    Args:
        change_metrics: 变化指标字典
        output_path: 可选的输出JSON路径
        
    Returns:
        自动生成的指令
    """
    generator = AutoPromptGenerator()
    prompts = generator.generate_auto_prompts(change_metrics)
    
    if output_path:
        generator.save_prompts_to_json(prompts, output_path)
    
    return prompts


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试变化指标
    test_metrics = {
        "volume_change": 30.5,
        "density_change": 75.2,
        "blur_change": 0.15,
        "is_new": True,
        "lesion_type": "磨玻璃"
    }
    
    generator = AutoPromptGenerator()
    prompts = generator.generate_auto_prompts(test_metrics)
    
    print("Auto-generated prompts:")
    for task_type, (positive, perturbed, reverse) in prompts.items():
        print(f"\nTask: {task_type}")
        print(f"Positive: {positive}")
        print(f"Perturbed: {perturbed}")
        print(f"Reverse: {reverse}")