#!/usr/bin/env python3
"""
最终简化版本的纵向推理分割测试脚本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalLongitudinalDataset(Dataset):
    """简化的纵向数据集"""
    def __init__(self, num_samples=20, num_timepoints=3, image_size=64):
        self.num_samples = num_samples
        self.num_timepoints = num_timepoints
        self.image_size = image_size
        
        # 生成模拟数据
        self.data = []
        for i in range(num_samples):
            sample_data = []
            for t in range(num_timepoints):
                # 模拟CT扫描图像 (1通道)
                image = torch.randn(1, image_size, image_size, image_size)
                # 模拟分割掩码 (1通道)
                mask = torch.randint(0, 2, (1, image_size, image_size, image_size)).float()
                sample_data.append({
                    'image': image,
                    'mask': mask,
                    'patient_id': f'patient_{i:03d}',
                    'timepoint': t,
                    'text_description': f'nodule size {10+t*2}mm'
                })
            self.data.append(sample_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class MinimalSiameseUNet(nn.Module):
    """极简Siamese UNet模型"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 编码器
        self.enc1 = self._make_conv_block(in_channels, base_channels)
        self.enc2 = self._make_conv_block(base_channels, base_channels * 2)
        self.enc3 = self._make_conv_block(base_channels * 2, base_channels * 4)
        
        # 解码器
        self.dec3 = self._make_conv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_conv_block(base_channels * 2, base_channels)
        self.dec1 = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1: 时间t1的图像 [B, C, H, W, D]
            x2: 时间t2的图像 [B, C, H, W, D]
        Returns:
            分割预测 [B, out_channels, H, W, D]
        """
        # 编码器路径
        enc1_1 = self.enc1(x1)
        enc1_2 = self.enc2(F.max_pool3d(enc1_1, 2))
        enc1_3 = self.enc3(F.max_pool3d(enc1_2, 2))
        
        enc2_1 = self.enc1(x2)
        enc2_2 = self.enc2(F.max_pool3d(enc2_1, 2))
        enc2_3 = self.enc3(F.max_pool3d(enc2_2, 2))
        
        # 简单的差分特征
        diff_features = torch.abs(enc1_3 - enc2_3)
        
        # 解码器路径 - 确保尺寸正确恢复
        dec3 = self.dec3(F.interpolate(diff_features, size=enc1_2.shape[2:], mode='trilinear', align_corners=False))
        dec2 = self.dec2(F.interpolate(dec3, size=enc1_1.shape[2:], mode='trilinear', align_corners=False))
        output = self.dec1(dec2)
        
        return torch.sigmoid(output)

class MinimalTrainer:
    """简化训练器"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_data in dataloader:
            # 处理每个样本
            for sample_data in batch_data:
                # 获取时间序列数据
                timepoints = len(sample_data)
                if timepoints < 2:
                    continue
                
                # 随机选择两个时间点
                t1_idx = 0
                t2_idx = min(1, timepoints - 1)
                
                # 获取数据
                image_t1 = sample_data[t1_idx]['image'].unsqueeze(0).to(self.device)
                image_t2 = sample_data[t2_idx]['image'].unsqueeze(0).to(self.device)
                mask_t2 = sample_data[t2_idx]['mask'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                pred_mask = self.model(image_t1, image_t2)
                
                # 确保尺寸匹配
                if pred_mask.shape != mask_t2.shape:
                    # 裁剪或填充到目标尺寸
                    if pred_mask.shape[2] > mask_t2.shape[2]:
                        # 裁剪
                        pred_mask = pred_mask[:, :, :mask_t2.shape[2], :mask_t2.shape[3], :mask_t2.shape[4]]
                    else:
                        # 填充
                        pad_dims = []
                        for i in range(2, 5):  # 空间维度
                            pad_dims.append(mask_t2.shape[i] - pred_mask.shape[i])
                            pad_dims.append(0)
                        pred_mask = F.pad(pred_mask, pad_dims[::-1])
                
                loss = self.criterion(pred_mask, mask_t2)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'loss': total_loss / max(num_batches, 1)}
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                for sample_data in batch_data:
                    timepoints = len(sample_data)
                    if timepoints < 2:
                        continue
                    
                    t1_idx = 0
                    t2_idx = min(1, timepoints - 1)
                    
                    image_t1 = sample_data[t1_idx]['image'].unsqueeze(0).to(self.device)
                    image_t2 = sample_data[t2_idx]['image'].unsqueeze(0).to(self.device)
                    mask_t2 = sample_data[t2_idx]['mask'].to(self.device)
                    
                    pred_mask = self.model(image_t1, image_t2)
                    
                    # 确保尺寸匹配
                    if pred_mask.shape != mask_t2.shape:
                        if pred_mask.shape[2] > mask_t2.shape[2]:
                            pred_mask = pred_mask[:, :, :mask_t2.shape[2], :mask_t2.shape[3], :mask_t2.shape[4]]
                        else:
                            pad_dims = []
                            for i in range(2, 5):
                                pad_dims.append(mask_t2.shape[i] - pred_mask.shape[i])
                                pad_dims.append(0)
                            pred_mask = F.pad(pred_mask, pad_dims[::-1])
                    
                    loss = self.criterion(pred_mask, mask_t2)
                    total_loss += loss.item()
                    num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}

def collate_fn(batch):
    """自定义collate函数"""
    return batch

def main():
    """主函数"""
    logger.info("Starting minimal longitudinal segmentation test...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = MinimalLongitudinalDataset(num_samples=20)
    val_dataset = MinimalLongitudinalDataset(num_samples=5)
    
    logger.info(f"Created training dataset with {len(train_dataset)} samples")
    logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    logger.info("Creating model...")
    model = MinimalSiameseUNet(in_channels=1, out_channels=1, base_channels=16)
    logger.info(f"Created MinimalSiameseUNet with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建训练器
    logger.info("Creating trainer...")
    trainer = MinimalTrainer(model, device=device)
    
    # 训练循环
    num_epochs = 5
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_dataloader)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # 验证
        val_metrics = trainer.validate(val_dataloader)
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
    
    logger.info("\nTraining completed successfully!")
    
    # 保存模型
    torch.save(model.state_dict(), 'minimal_siamese_unet.pth')
    logger.info("Model saved to minimal_siamese_unet.pth")
    
    # 简单推理测试
    logger.info("\nRunning inference test...")
    model.eval()
    with torch.no_grad():
        sample_data = train_dataset[0]
        image_t1 = sample_data[0]['image'].unsqueeze(0).to(device)
        image_t2 = sample_data[1]['image'].unsqueeze(0).to(device)
        
        pred_mask = model(image_t1, image_t2)
        logger.info(f"Inference successful! Prediction shape: {pred_mask.shape}")
        logger.info(f"Prediction range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")

if __name__ == "__main__":
    main()