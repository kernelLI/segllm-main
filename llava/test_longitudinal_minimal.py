"""
LIDC-IDRI纵向推理分割最小测试脚本
简化版本，避免复杂的依赖冲突
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalLongitudinalDataset(Dataset):
    """最小纵向推理分割数据集"""
    
    def __init__(self, num_samples: int = 20, image_size: Tuple[int, int, int] = (64, 64, 32)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.instructions = [
            "请识别体积增大的结节区域",
            "请检测新出现的病灶",
            "请识别密度变化的区域",
            "请分割形态学变化的结节"
        ]
        
        logger.info(f"Created minimal dataset with {num_samples} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 生成模拟数据
        ct_t0 = torch.randn(1, *self.image_size)
        ct_t1 = torch.randn(1, *self.image_size)
        
        # 生成模拟目标mask
        target_mask = torch.zeros(*self.image_size)
        # 在中心区域创建一些变化
        center = [s // 2 for s in self.image_size]
        target_mask[
            center[0]-8:center[0]+8,
            center[1]-8:center[1]+8,
            center[2]-4:center[2]+4
        ] = 1.0
        
        instruction = self.instructions[idx % len(self.instructions)]
        
        return {
            "ct_t0": ct_t0,
            "ct_t1": ct_t1,
            "instruction": instruction,
            "target_mask": target_mask,
            "sample_id": f"sample_{idx}"
        }

class MinimalSiameseUNet(nn.Module):
    """最小Siamese UNet模型"""
    
    def __init__(self, input_channels: int = 1, output_channels: int = 1, base_channels: int = 16):
        super().__init__()
        
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 差分特征提取
        self.diff_conv = nn.Conv3d(base_channels * 4, base_channels * 2, 3, padding=1)
        
        # 解码器
        self.decoder2 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, output_channels, 3, padding=1)
        )
        
        logger.info(f"Created MinimalSiameseUNet with base_channels={base_channels}")
    
    def forward(self, ct_t0: torch.Tensor, ct_t1: torch.Tensor, text_features: torch.Tensor = None) -> torch.Tensor:
        # 编码器
        enc1_t0 = self.encoder1(ct_t0)
        enc1_t1 = self.encoder1(ct_t1)
        
        enc2_t0 = self.encoder2(enc1_t0)
        enc2_t1 = self.encoder2(enc1_t1)
        
        # 特征融合（拼接）
        fused_features = torch.cat([enc2_t0, enc2_t1], dim=1)
        
        # 差分特征提取
        diff_features = self.diff_conv(fused_features)
        
        # 解码器
        dec2 = self.decoder2(diff_features)
        
        # 调整尺寸以匹配跳跃连接
        if dec2.shape != enc1_t0.shape:
            dec2 = F.interpolate(dec2, size=enc1_t0.shape[2:], mode='trilinear', align_corners=False)
        
        output = self.decoder1(dec2)
        
        return output

class MinimalTrainer:
    """最小训练器"""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        
        logger.info(f"Created MinimalTrainer on device: {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动到设备
            ct_t0 = batch["ct_t0"].to(self.device)
            ct_t1 = batch["ct_t1"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            
            # 前向传播
            pred_mask = self.model(ct_t0, ct_t1)
            
            # 计算损失
            loss = self.criterion(pred_mask, target_mask)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动到设备
                ct_t0 = batch["ct_t0"].to(self.device)
                ct_t1 = batch["ct_t1"].to(self.device)
                target_mask = batch["target_mask"].to(self.device)
                
                # 前向传播
                pred_mask = self.model(ct_t0, ct_t1)
                
                # 计算损失
                loss = self.criterion(pred_mask, target_mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def predict_sample(self, ct_t0: torch.Tensor, ct_t1: torch.Tensor) -> torch.Tensor:
        """预测单个样本"""
        self.model.eval()
        with torch.no_grad():
            pred_mask = self.model(ct_t0.unsqueeze(0), ct_t1.unsqueeze(0))
            return torch.sigmoid(pred_mask.squeeze(0))

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算Dice系数"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    
    if union == 0:
        return 1.0
    
    return (2.0 * intersection / union).item()

def main():
    """主函数"""
    
    logger.info("Starting minimal longitudinal segmentation test...")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = MinimalLongitudinalDataset(num_samples=20)
    val_dataset = MinimalLongitudinalDataset(num_samples=5)
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # 创建模型
    logger.info("Creating model...")
    model = MinimalSiameseUNet(input_channels=1, output_channels=1, base_channels=16)
    
    # 创建训练器
    logger.info("Creating trainer...")
    trainer = MinimalTrainer(model, device)
    
    # 训练循环
    logger.info("Starting training...")
    num_epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f"Train metrics: {train_metrics}")
        
        # 验证
        val_metrics = trainer.validate(val_dataloader)
        logger.info(f"Val metrics: {val_metrics}")
        
        # 保存最佳模型
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
    
    # 测试预测
    logger.info("\nTesting prediction...")
    test_sample = val_dataset[0]
    ct_t0, ct_t1, target_mask = test_sample["ct_t0"], test_sample["ct_t1"], test_sample["target_mask"]
    
    pred_mask = trainer.predict_sample(ct_t0, ct_t1)
    
    # 计算指标
    dice_score = calculate_dice(pred_mask, target_mask)
    logger.info(f"Test sample dice score: {dice_score:.4f}")
    
    # 保存结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "num_epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "final_train_loss": train_metrics["loss"],
        "final_val_loss": val_metrics["val_loss"],
        "test_dice_score": dice_score,
        "model_params": sum(p.numel() for p in model.parameters())
    }
    
    results_path = "minimal_longitudinal_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Test completed! Results saved to {results_path}")
    logger.info(f"Model parameters: {results['model_params']:,}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Test dice score: {dice_score:.4f}")

if __name__ == "__main__":
    main()