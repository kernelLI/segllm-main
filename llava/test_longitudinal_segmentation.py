"""
LIDC-IDRI纵向推理分割完整测试脚本
集成所有模块，基于SegLLM架构进行端到端测试
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from llava.data.lidc_longitudinal_data_generator import (
    LIDCLongitudinalDataGenerator, create_lidc_longitudinal_dataset
)
from llava.data.longitudinal_registration import (
    LongitudinalRegistrationPipeline, create_longitudinal_registration_pipeline
)
from llava.data.longitudinal_changes import (
    LongitudinalChangeCalculator, create_longitudinal_change_calculator
)
from llava.data.longitudinal_instruction_generator import (
    LongitudinalInstructionGenerator, create_longitudinal_instruction_generator
)
from llava.model.siamese_unet_longitudinal import (
    SiameseUNetForLongitudinalSegmentation, create_siamese_unet_for_longitudinal_segmentation
)
from llava.model.text_conditioning_longitudinal import (
    LongitudinalTextConditioning, create_longitudinal_text_conditioning
)
from llava.model.longitudinal_metrics import (
    LongitudinalSegmentationMetrics, compute_longitudinal_metrics
)

logger = logging.getLogger(__name__)

class LIDCLongitudinalDataset(Dataset):
    """LIDC-IDRI纵向推理分割数据集"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        max_samples: int = 100,
        instruction_types: List[str] = None,
        language: str = "zh"
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.instruction_types = instruction_types or ["volume_threshold", "new_lesion", "density_change"]
        self.language = language
        
        # 初始化数据生成器
        self.data_generator = LIDCLongitudinalDataGenerator(data_dir)
        self.instruction_generator = LongitudinalInstructionGenerator(language=language)
        
        # 加载数据
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本"""
        
        # 生成纵向配对数据
        longitudinal_pairs = self.data_generator.find_longitudinal_pairs()
        
        samples = []
        for i, pair in enumerate(longitudinal_pairs[:self.max_samples]):
            # 计算变化
            change_metrics = self.data_generator.calculate_changes(pair)
            
            # 生成指令-标签对
            instruction_pairs = self.instruction_generator.generate_instruction_label_pairs(
                change_metrics, self.instruction_types
            )
            
            for instruction_pair in instruction_pairs:
                sample = {
                    "pair_id": f"pair_{i}",
                    "ct_t0_path": pair["ct_t0_path"],
                    "ct_t1_path": pair["ct_t1_path"],
                    "mask_t0_path": pair["mask_t0_path"],
                    "mask_t1_path": pair["mask_t1_path"],
                    "instruction": instruction_pair["instruction"],
                    "target_mask": instruction_pair["target_mask"],
                    "change_metrics": change_metrics,
                    "task_type": instruction_pair["task_type"]
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 这里简化处理，实际应该加载真实的CT和mask数据
        # 使用随机数据作为示例
        ct_shape = (1, 64, 64, 32)  # 单通道3D CT
        mask_shape = (64, 64, 32)   # 3D mask
        
        ct_t0 = torch.randn(*ct_shape)
        ct_t1 = torch.randn(*ct_shape)
        mask_t0 = (torch.randn(*mask_shape) > 0).float()
        mask_t1 = (torch.randn(*mask_shape) > 0).float()
        target_mask = (torch.randn(*mask_shape) > 0).float()
        
        # 根据任务类型调整目标mask
        if sample["task_type"] == "volume_threshold":
            # 模拟体积变化阈值任务
            target_mask = torch.where(torch.randn(*mask_shape) > 0.3, 1.0, 0.0)
        elif sample["task_type"] == "new_lesion":
            # 模拟新病灶任务
            target_mask = torch.where(torch.randn(*mask_shape) > 0.7, 1.0, 0.0)
        elif sample["task_type"] == "density_change":
            # 模拟密度变化任务
            target_mask = torch.where(torch.randn(*mask_shape) > 0.5, 1.0, 0.0)
        
        return {
            "ct_t0": ct_t0,
            "ct_t1": ct_t1,
            "mask_t0": mask_t0,
            "mask_t1": mask_t1,
            "instruction": sample["instruction"],
            "target_mask": target_mask,
            "change_metrics": sample["change_metrics"],
            "task_type": sample["task_type"],
            "pair_id": sample["pair_id"]
        }

class LongitudinalSegmentationTrainer:
    """纵向推理分割训练器"""
    
    def __init__(
        self,
        model: SiameseUNetForLongitudinalSegmentation,
        text_conditioning: LongitudinalTextConditioning,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.text_conditioning = text_conditioning.to(device)
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(text_conditioning.parameters()),
            lr=1e-4, weight_decay=1e-5
        )
        
        # 损失函数
        self.dice_loss = self._dice_loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 指标计算器
        self.metrics_calculator = LongitudinalSegmentationMetrics(
            voxel_spacing=(1.0, 1.0, 2.0), device=device
        )
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def _dice_loss(self, pred: Tensor, target: Tensor, smooth: float = 1e-5) -> Tensor:
        """Dice损失"""
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2.0 * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)
        return 1.0 - dice
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        self.text_conditioning.train()
        
        total_loss = 0.0
        total_dice_loss = 0.0
        total_bce_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动到设备
            ct_t0 = batch["ct_t0"].to(self.device)
            ct_t1 = batch["ct_t1"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            instructions = batch["instruction"]
            
            # 编码文本指令
            with torch.set_grad_enabled(False):
                text_results = self.text_conditioning.text_encoder(instructions)
                text_embeddings = text_results["text_embeddings"]
            
            # 前向传播
            pred_mask = self.model(ct_t0, ct_t1, text_embeddings)
            
            # 计算损失
            dice_loss = self.dice_loss(pred_mask, target_mask)
            bce_loss = self.bce_loss(pred_mask, target_mask)
            total_batch_loss = dice_loss + bce_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            # 更新指标
            with torch.no_grad():
                self.metrics_calculator.update(
                    torch.sigmoid(pred_mask), target_mask, instructions[0], batch["change_metrics"][0]
                )
            
            # 累计损失
            total_loss += total_batch_loss.item()
            total_dice_loss += dice_loss.item()
            total_bce_loss += bce_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {total_batch_loss.item():.4f}")
        
        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        avg_dice_loss = total_dice_loss / num_batches
        avg_bce_loss = total_bce_loss / num_batches
        metrics = self.metrics_calculator.compute()
        
        return {
            "loss": avg_loss,
            "dice_loss": avg_dice_loss,
            "bce_loss": avg_bce_loss,
            **metrics
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        
        self.model.eval()
        self.text_conditioning.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                ct_t0 = batch["ct_t0"].to(self.device)
                ct_t1 = batch["ct_t1"].to(self.device)
                target_mask = batch["target_mask"].to(self.device)
                instructions = batch["instruction"]
                
                # 编码文本指令
                text_results = self.text_conditioning.text_encoder(instructions)
                text_embeddings = text_results["text_embeddings"]
                
                # 前向传播
                pred_mask = self.model(ct_t0, ct_t1, text_embeddings)
                
                # 计算损失
                dice_loss = self.dice_loss(pred_mask, target_mask)
                bce_loss = self.bce_loss(pred_mask, target_mask)
                total_batch_loss = dice_loss + bce_loss
                
                # 更新指标
                self.metrics_calculator.update(
                    torch.sigmoid(pred_mask), target_mask, instructions[0], batch["change_metrics"][0]
                )
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        # 计算平均损失和指标
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calculator.compute()
        
        return {
            "val_loss": avg_loss,
            **metrics
        }
    
    def save_checkpoint(self, epoch: int, save_dir: str, metrics: Dict[str, float]):
        """保存检查点"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "text_conditioning_state_dict": self.text_conditioning.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # 也保存最佳模型
        if "dice_mean" in metrics and metrics["dice_mean"] > 0.8:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="LIDC-IDRI纵向推理分割测试")
    parser.add_argument("--data_dir", type=str, default="./data/lidc_idri", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./outputs/longitudinal", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--max_samples", type=int, default=50, help="最大样本数")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"], help="语言")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 设置日志
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting LIDC-IDRI longitudinal segmentation test")
    logger.info(f"Device: {device}")
    logger.info(f"Arguments: {args}")
    
    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = LIDCLongitudinalDataset(
        data_dir=args.data_dir,
        split="train",
        max_samples=args.max_samples,
        language=args.language
    )
    
    val_dataset = LIDCLongitudinalDataset(
        data_dir=args.data_dir,
        split="val",
        max_samples=args.max_samples // 4,
        language=args.language
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    logger.info("Creating models...")
    model = create_siamese_unet_for_longitudinal_segmentation(
        input_channels=1,
        output_channels=1,
        base_channels=32,
        depth=3,
        text_embedding_dim=768,
        difference_method="concat",
        fusion_method="film"
    )
    
    text_conditioning = create_longitudinal_text_conditioning(
        text_encoder_name="bert-base-chinese" if args.language == "zh" else "bert-base-uncased",
        image_channels=32 * (2 ** 2),  # 根据模型深度调整
        modulation_type="film",
        feature_types=["volume", "density", "morphology", "change_type"]
    )
    
    # 创建训练器
    logger.info("Creating trainer...")
    trainer = LongitudinalSegmentationTrainer(model, text_conditioning, device)
    
    # 训练循环
    logger.info("Starting training...")
    best_val_dice = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f"Train metrics: {train_metrics}")
        
        # 验证
        val_metrics = trainer.validate(val_dataloader)
        logger.info(f"Val metrics: {val_metrics}")
        
        # 保存检查点
        trainer.save_checkpoint(epoch, args.output_dir, {**train_metrics, **val_metrics})
        
        # 更新最佳模型
        if val_metrics.get("dice_mean", 0) > best_val_dice:
            best_val_dice = val_metrics["dice_mean"]
            logger.info(f"New best validation dice: {best_val_dice:.4f}")
    
    # 最终测试
    logger.info("\nFinal testing...")
    test_dataset = LIDCLongitudinalDataset(
        data_dir=args.data_dir,
        split="test",
        max_samples=args.max_samples // 4,
        language=args.language
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_metrics = trainer.validate(test_dataloader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # 保存最终结果
    results = {
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_val_dice": best_val_dice,
        "timestamp": datetime.now().isoformat()
    }
    
    results_path = os.path.join(args.output_dir, "final_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test completed! Results saved to {results_path}")
    logger.info(f"Best validation dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()