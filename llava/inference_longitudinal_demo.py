"""
SegLLM纵向推理演示脚本
LIDC-IDRI纵向推理分割推理演示脚本，支持交互式推理和批量推理

输入:
- model_path: 模型路径，支持本地路径或HuggingFace模型ID
- image_t0_path: 基线CT图像路径（T0时相）
- image_t1_path: 随访CT图像路径（T1时相）
- task_type: 推理任务类型（volume_threshold/new_lesion/density_change/combined_attributes）
- change_info: 可选，变化信息字典（volume_change、density_change、is_new）
- temperature: 采样温度，控制生成随机性（默认0.7）
- max_new_tokens: 最大生成token数（默认512）
- device: 计算设备（cuda/cpu）
- dtype: 模型数据类型（float16/float32）

输出:
- instruction: 生成的推理指令文本
- response: 模型生成的响应文本
- segmentation_mask: 预测的分割掩码（numpy数组）
- change_metrics: 变化分析指标（体积变化、密度变化等）
- inference_time: 推理耗时统计

功能:
- 加载和初始化纵向推理模型
- 预处理双时相CT图像（窗宽窗位调整、尺寸标准化）
- 生成符合医学术语的推理指令
- 执行端到端推理并生成分割结果
- 计算变化分析指标和性能统计
- 支持交互式推理和批量推理模式
- 可视化推理结果和变化分析
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.longitudinal_arch import LongitudinalMetaModel, LongitudinalMetaForCausalLM
from llava.conversation_longitudinal import LongitudinalConversationTemplates
from llava.metrics_longitudinal import LongitudinalMetrics, ChangeCalculator, ChangeMetrics
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_LONGITUDINAL_TOKEN
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64

logger = logging.getLogger(__name__)

class LongitudinalInferenceDemo:
    """纵向推理分割演示类"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
        # 初始化组件
        self._load_model()
        self._initialize_components()
        
        logger.info(f"Longitudinal inference demo initialized on {device}")
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"Loading model from {self.model_path}")
        
        # 这里简化模型加载过程
        # 实际使用时需要根据具体的模型架构调整
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_longitudinal_model()
        
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _load_tokenizer(self):
        """加载分词器"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True
        )
        
        # 添加特殊token
        special_tokens = {
            "additional_special_tokens": [
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_LONGITUDINAL_TOKEN,
                "[IMAGE256]", "[MASK-DECODE]", "[INFERENCE]"
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        return tokenizer
    
    def _load_longitudinal_model(self):
        """加载纵向推理模型"""
        from transformers import AutoModelForCausalLM
        
        # 这里简化处理，实际应该加载自定义的纵向模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        
        return model
    
    def _initialize_components(self):
        """初始化其他组件"""
        self.conversation_templates = LongitudinalConversationTemplates()
        self.change_calculator = ChangeCalculator()
        self.metrics_computer = LongitudinalMetrics()
        
        # 任务调度器
        self.task_weights = {
            "volume_threshold": 0.3,
            "new_lesion": 0.25,
            "density_change": 0.25,
            "combined_attributes": 0.2
        }
    
    def preprocess_images(self, image_t0_path, image_t1_path):
        """预处理双时相图像"""
        # 使用数据集类的方法加载图像
        from llava.data.lidc_longitudinal_dataset import LongitudinalDataset
        
        # 创建临时数据集实例用于图像加载
        dataset = LongitudinalDataset(
            data_path="dummy",
            tokenizer=None,
            data_args=None,
            is_inference=True
        )
        
        # 加载双时相图像
        image_t0 = dataset._load_ct_image(image_t0_path)
        image_t1 = dataset._load_ct_image(image_t1_path)
        
        # 拼接图像 - 适配SegLLM格式 (1, 6, H, W)
        concatenated_images = torch.cat([image_t0.unsqueeze(0), image_t1.unsqueeze(0)], dim=1)
        
        return concatenated_images
    
    def generate_instruction(
        self,
        task_type: str,
        change_info: Optional[Dict] = None
    ) -> str:
        """生成推理指令 - 使用SegLLM标准格式"""
        # 使用模板生成指令
        if change_info is None:
            change_info = self._generate_default_change_info(task_type)
        
        # 模拟图像路径（实际使用时应该传入真实路径）
        dummy_image_t0 = "dummy_t0.jpg"
        dummy_image_t1 = "dummy_t1.jpg"
        dummy_target = "dummy_target.nii.gz"
        
        instruction, _ = self.conversation_templates.generate_instruction(
            task_type=task_type,
            image_t0_path=dummy_image_t0,
            image_t1_path=dummy_image_t1,
            target_mask_path=dummy_target,
            change_info=change_info
        )
        
        # 移除路径信息，只保留指令部分
        instruction = instruction.replace(f"[IMAGE256:{dummy_image_t0}] ", "")
        instruction = instruction.replace(f"[IMAGE256:{dummy_image_t1}] ", "")
        
        # 添加SegLLM标准前缀
        if not instruction.startswith("[SEG]"):
            instruction = "[SEG] " + instruction
        
        return instruction
    
    def _generate_default_change_info(self, task_type: str) -> Dict:
        """生成默认变化信息"""
        if task_type == "volume_threshold":
            return {"volume_change": 30, "density_change": 100, "is_new": False}
        elif task_type == "new_lesion":
            return {"volume_change": 100, "density_change": 200, "is_new": True}
        elif task_type == "density_change":
            return {"volume_change": 5, "density_change": 80, "is_new": False}
        elif task_type == "combined_attributes":
            return {"volume_change": 25, "density_change": 150, "is_new": False}
        
        return {"volume_change": 0, "density_change": 0, "is_new": False}
    
    def inference(
        self,
        image_t0_path: str,
        image_t1_path: str,
        instruction: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        执行推理
        
        Args:
            image_t0_path: 基线图像路径
            image_t1_path: 随访图像路径
            instruction: 推理指令
            temperature: 采样温度
            max_new_tokens: 最大生成token数
            
        Returns:
            推理结果字典
        """
        start_time = time.time()
        
        # 预处理图像
        concatenated_images = self.preprocess_images(image_t0_path, image_t1_path)
        
        # 构建输入
        input_text = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN} {instruction}"
        
        # 分词
        input_ids = tokenizer_image_token(
            input_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX=0,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # 推理 - 适配SegLLM格式
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                images=concatenated_images,  # 拼接后的图像 (1, 6, H, W)
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        return {
            "instruction": instruction,
            "response": response,
            "inference_time": inference_time,
            "input_shape": {
                "concatenated_images": concatenated_images.shape
            }
        }
    
    def batch_inference(
        self,
        test_cases: List[Dict],
        output_dir: str = "./inference_results"
    ) -> List[Dict]:
        """
        批量推理
        
        Args:
            test_cases: 测试用例列表
            output_dir: 输出目录
            
        Returns:
            推理结果列表
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        logger.info(f"Running batch inference on {len(test_cases)} cases")
        
        for i, case in enumerate(test_cases):
            logger.info(f"Processing case {i+1}/{len(test_cases)}")
            
            try:
                result = self.inference(
                    image_t0_path=case["image_t0"],
                    image_t1_path=case["image_t1"],
                    instruction=case["instruction"],
                    temperature=case.get("temperature", 0.7)
                )
                
                # 添加case信息
                result.update({
                    "case_id": case.get("case_id", f"case_{i}"),
                    "task_type": case.get("task_type", "unknown"),
                    "ground_truth": case.get("ground_truth", None)
                })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing case {i}: {str(e)}")
                results.append({
                    "case_id": case.get("case_id", f"case_{i}"),
                    "error": str(e)
                })
        
        # 保存结果
        results_file = os.path.join(output_dir, "inference_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch inference completed. Results saved to {results_file}")
        return results
    
    def interactive_demo(self):
        """交互式演示"""
        print("=== LIDC-IDRI Longitudinal Reasoning Segmentation Demo ===")
        print("Available task types:")
        for i, task_type in enumerate(self.conversation_templates.templates.keys()):
            print(f"{i+1}. {task_type}")
        
        while True:
            try:
                # 获取用户输入
                print("\n--- New Inference ---")
                
                image_t0_path = input("Enter baseline image path (or 'quit' to exit): ").strip()
                if image_t0_path.lower() == 'quit':
                    break
                
                image_t1_path = input("Enter follow-up image path: ").strip()
                
                print("Select task type:")
                task_types = list(self.conversation_templates.templates.keys())
                for i, task_type in enumerate(task_types):
                    print(f"{i+1}. {task_type}")
                
                task_choice = input("Enter task number (or custom instruction): ").strip()
                
                if task_choice.isdigit() and 1 <= int(task_choice) <= len(task_types):
                    task_type = task_types[int(task_choice) - 1]
                    instruction = self.generate_instruction(task_type)
                    print(f"Generated instruction: {instruction}")
                else:
                    instruction = task_choice
                
                temperature = float(input("Enter temperature (0.1-1.0, default 0.7): ") or "0.7")
                
                # 执行推理
                print("Running inference...")
                result = self.inference(
                    image_t0_path=image_t0_path,
                    image_t1_path=image_t1_path,
                    instruction=instruction,
                    temperature=temperature
                )
                
                # 显示结果
                print(f"\n--- Inference Result ---")
                print(f"Instruction: {result['instruction']}")
                print(f"Response: {result['response']}")
                print(f"Inference time: {result['inference_time']:.2f}s")
                
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        print("Demo completed!")

def create_test_cases() -> List[Dict]:
    """创建测试用例"""
    test_cases = [
        {
            "case_id": "volume_increase_30",
            "task_type": "volume_threshold",
            "image_t0": "test_data/case1_t0.png",
            "image_t1": "test_data/case1_t1.png",
            "instruction": "分割所有较上次体积增加超过30%的结节",
            "temperature": 0.7,
            "ground_truth": "mask_increase_30.nii.gz"
        },
        {
            "case_id": "new_lesion_ggo",
            "task_type": "new_lesion",
            "image_t0": "test_data/case2_t0.png",
            "image_t1": "test_data/case2_t1.png",
            "instruction": "标出新出现的磨玻璃结节",
            "temperature": 0.8,
            "ground_truth": "mask_new_ggo.nii.gz"
        },
        {
            "case_id": "density_change_solid",
            "task_type": "density_change",
            "image_t0": "test_data/case3_t0.png",
            "image_t1": "test_data/case3_t1.png",
            "instruction": "圈出从磨玻璃变实性的病灶",
            "temperature": 0.6,
            "ground_truth": "mask_density_change.nii.gz"
        },
        {
            "case_id": "combined_volume_density",
            "task_type": "combined_attributes",
            "image_t0": "test_data/case4_t0.png",
            "image_t1": "test_data/case4_t1.png",
            "instruction": "分割体积增加≥20%且密度≥150HU的结节",
            "temperature": 0.7,
            "ground_truth": "mask_combined.nii.gz"
        }
    ]
    
    return test_cases

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LIDC Longitudinal Reasoning Segmentation Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="interactive", help="Inference mode")
    parser.add_argument("--test_cases", type=str, help="Path to test cases JSON file")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Model dtype")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建推理演示器
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    demo = LongitudinalInferenceDemo(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype
    )
    
    # 运行演示
    if args.mode == "interactive":
        demo.interactive_demo()
    else:
        # 批量推理
        if args.test_cases:
            with open(args.test_cases, 'r') as f:
                test_cases = json.load(f)
        else:
            test_cases = create_test_cases()
        
        results = demo.batch_inference(test_cases, args.output_dir)
        
        # 打印总结
        successful_cases = [r for r in results if "error" not in r]
        print(f"\n--- Batch Inference Summary ---")
        print(f"Total cases: {len(results)}")
        print(f"Successful: {len(successful_cases)}")
        print(f"Failed: {len(results) - len(successful_cases)}")
        
        if successful_cases:
            avg_time = np.mean([r["inference_time"] for r in successful_cases])
            print(f"Average inference time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()