#!/usr/bin/env python3
"""
并行式纵向推理分割训练脚本 - 符合AI项目训练常态
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import os
import sys
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    autocast = None
    GradScaler = None
import multiprocessing
from functools import partial
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelLongitudinalDataset(Dataset):
    """并行化纵向数据集"""
    def __init__(self, num_samples=100, num_timepoints=3, image_size=64, cache_size=1000):
        self.num_samples = num_samples
        self.num_timepoints = num_timepoints
        self.image_size = image_size
        self.cache_size = cache_size
        
        # 预生成数据以提高效率
        self._pre_generate_data()
    
    def _pre_generate_data(self):
        """预生成数据以提高加载效率"""
        logger.info(f"Pre-generating {self.num_samples} samples...")
        start_time = time.time()
        
        self.data = []
        for i in range(self.num_samples):
            sample_data = []
            base_image = torch.randn(1, self.image_size, self.image_size, self.image_size)
            base_mask = torch.randint(0, 2, (1, self.image_size, self.image_size, self.image_size)).float()
            
            for t in range(self.num_timepoints):
                # 添加时间变化
                noise_scale = 0.1 * t
                image = base_image + noise_scale * torch.randn_like(base_image)
                mask = base_mask.clone()
                # 模拟结节增长
                if t > 0:
                    mask = torch.clamp(mask + 0.1 * t * torch.rand_like(mask), 0, 1)
                
                sample_data.append({
                    'image': image,
                    'mask': mask,
                    'patient_id': f'patient_{i:03d}',
                    'timepoint': t,
                    'text_description': f'nodule size {10+t*2}mm'
                })
            self.data.append(sample_data)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data pre-generation completed in {elapsed_time:.2f} seconds")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class OptimizedSiameseUNet(nn.Module):
    """优化的Siamese UNet模型 - 支持混合精度训练"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 编码器 - 增加通道数以提高性能
        self.enc1 = self._make_conv_block(in_channels, base_channels)
        self.enc2 = self._make_conv_block(base_channels, base_channels * 2)
        self.enc3 = self._make_conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_conv_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.dec4 = self._make_conv_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._make_conv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_conv_block(base_channels * 2, base_channels)
        self.dec1 = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
        # Dropout用于正则化
        self.dropout = nn.Dropout3d(0.1)
        
    def _make_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
    
    def forward(self, x1, x2):
        """
        前向传播 - 支持混合精度
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
        enc1_4 = self.enc4(F.max_pool3d(enc1_3, 2))
        
        enc2_1 = self.enc1(x2)
        enc2_2 = self.enc2(F.max_pool3d(enc2_1, 2))
        enc2_3 = self.enc3(F.max_pool3d(enc2_2, 2))
        enc2_4 = self.enc4(F.max_pool3d(enc2_3, 2))
        
        # 差分特征
        diff_features = torch.abs(enc1_4 - enc2_4)
        
        # 解码器路径
        dec4 = self.dec4(F.interpolate(diff_features, size=enc1_3.shape[2:], mode='trilinear', align_corners=True))
        dec3 = self.dec3(F.interpolate(dec4, size=enc1_2.shape[2:], mode='trilinear', align_corners=True))
        dec2 = self.dec2(F.interpolate(dec3, size=enc1_1.shape[2:], mode='trilinear', align_corners=True))
        output = self.dec1(dec2)
        
        return torch.sigmoid(output)

class ParallelTrainer:
    """并行化训练器 - 支持混合精度和梯度累积"""
    def __init__(self, model, device='cuda', accumulation_steps=4, use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available() and AMP_AVAILABLE
        
        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 混合精度训练
        self.scaler = GradScaler() if self.use_amp else None
        
        # 训练统计
        self.global_step = 0
        self.epoch_losses = []
    
    def process_batch(self, batch_data):
        """处理批次数据"""
        batch_images_t1 = []
        batch_images_t2 = []
        batch_masks = []
        
        for sample_data in batch_data:
            timepoints = len(sample_data)
            if timepoints < 2:
                continue
            
            # 选择两个时间点
            t1_idx = 0
            t2_idx = min(1, timepoints - 1)
            
            # 收集数据
            batch_images_t1.append(sample_data[t1_idx]['image'])
            batch_images_t2.append(sample_data[t2_idx]['image'])
            batch_masks.append(sample_data[t2_idx]['mask'])
        
        if not batch_images_t1:
            return None, None, None
        
        # 堆叠成批次
        images_t1 = torch.stack(batch_images_t1).to(self.device)
        images_t2 = torch.stack(batch_images_t2).to(self.device)
        masks = torch.stack(batch_masks).to(self.device)
        
        return images_t1, images_t2, masks
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch - 支持混合精度"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        running_loss = 0.0
        
        progress_interval = max(1, len(dataloader) // 10)  # 每10%显示一次进度
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 处理批次数据
            images_t1, images_t2, masks = self.process_batch(batch_data)
            if images_t1 is None:
                continue
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    pred_masks = self.model(images_t1, images_t2)
                    loss = self.criterion(pred_masks, masks)
                    loss = loss / self.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                pred_masks = self.model(images_t1, images_t2)
                loss = self.criterion(pred_masks, masks)
                
                # 反向传播
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # 统计
            total_loss += loss.item() * self.accumulation_steps
            running_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # 显示进度
            if (batch_idx + 1) % progress_interval == 0:
                avg_loss = running_loss / progress_interval
                logger.info(f"Epoch {epoch} - Batch {batch_idx + 1}/{len(dataloader)} - Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        # 更新学习率
        self.scheduler.step()
        
        return {'loss': total_loss / max(num_batches, 1), 'lr': self.scheduler.get_last_lr()[0]}
    
    def validate(self, dataloader, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                images_t1, images_t2, masks = self.process_batch(batch_data)
                if images_t1 is None:
                    continue
                
                if self.use_amp:
                    with autocast():
                        pred_masks = self.model(images_t1, images_t2)
                        loss = self.criterion(pred_masks, masks)
                else:
                    pred_masks = self.model(images_t1, images_t2)
                    loss = self.criterion(pred_masks, masks)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    def save_checkpoint(self, epoch, path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch_losses': self.epoch_losses
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

def collate_fn(batch):
    """自定义collate函数 - 保持原始结构"""
    return batch

def get_dataloader_kwargs():
    """获取数据加载器的推荐参数"""
    num_workers = min(4, multiprocessing.cpu_count())  # 使用CPU核心数，最多4个
    kwargs = {
        'batch_size': 4,  # 批处理大小
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': True,  # GPU训练时启用
        'collate_fn': collate_fn
    }
    
    # PyTorch 1.4.0不支持persistent_workers和prefetch_factor参数
    if torch.__version__ >= '1.7.0':
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    
    return kwargs

def main():
    """主函数 - 并行化训练"""
    logger.info("Starting parallel longitudinal segmentation training...")
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建数据集
    logger.info("Creating datasets...")
    train_dataset = ParallelLongitudinalDataset(num_samples=200)  # 增加样本数量
    val_dataset = ParallelLongitudinalDataset(num_samples=50)
    
    logger.info(f"Created training dataset with {len(train_dataset)} samples")
    logger.info(f"Created validation dataset with {len(val_dataset)} samples")
    
    # 创建数据加载器
    dataloader_kwargs = get_dataloader_kwargs()
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    
    # 验证集不需要shuffle和多个worker
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs['shuffle'] = False
    val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)
    
    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Val dataloader: {len(val_dataloader)} batches")
    
    # 创建模型
    logger.info("Creating model...")
    model = OptimizedSiameseUNet(in_channels=1, out_channels=1, base_channels=32)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created OptimizedSiameseUNet with {total_params:,} parameters")
    
    # 创建训练器
    logger.info("Creating parallel trainer...")
    trainer = ParallelTrainer(model, device=device, accumulation_steps=2, use_amp=True)
    
    # 训练循环
    num_epochs = 10  # 增加epoch数量
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_dataloader, epoch)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f} - LR: {train_metrics['lr']:.6f}")
        
        # 验证
        val_metrics = trainer.validate(val_dataloader, epoch)
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # 保存最佳模型
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            trainer.save_checkpoint(epoch, 'best_model_parallel.pth')
            logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch, f'checkpoint_epoch_{epoch + 1}.pth')
        
        # 记录epoch时间
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        logger.info(f"Epoch time: {epoch_time:.1f}s - Total time: {total_time/60:.1f}min")
        
        # 保存epoch损失
        trainer.epoch_losses.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['val_loss']
        })
    
    logger.info(f"\n{'='*50}")
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
    
    # 最终推理测试
    logger.info("\nRunning final inference test...")
    model.eval()
    with torch.no_grad():
        sample_data = train_dataset[0]
        image_t1 = sample_data[0]['image'].unsqueeze(0).to(device)
        image_t2 = sample_data[1]['image'].unsqueeze(0).to(device)
        
        if trainer.use_amp:
            with autocast():
                pred_mask = model(image_t1, image_t2)
        else:
            pred_mask = model(image_t1, image_t2)
        
        logger.info(f"Inference successful! Prediction shape: {pred_mask.shape}")
        logger.info(f"Prediction range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")

if __name__ == "__main__":
    main()