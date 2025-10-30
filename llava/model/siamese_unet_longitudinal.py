"""
Siamese UNet模型架构用于LIDC-IDRI纵向推理分割
基于SegLLM架构扩展，支持多时相CT图像的差分编码和文本条件调制
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from llava.model.longitudinal_arch import LongitudinalMetaModel, LongitudinalMetaForCausalLM

logger = logging.getLogger(__name__)

class SiameseUNetEncoder(nn.Module):
    """Siamese UNet编码器 - 共享权重的双分支编码"""
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # 构建编码器层
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # 第一层
        in_channels = input_channels
        out_channels = base_channels
        
        for i in range(depth):
            # 编码块
            encoder_block = self._make_encoder_block(
                in_channels, out_channels, dropout_rate, use_batch_norm
            )
            self.encoder_layers.append(encoder_block)
            
            # 下采样层
            if i < depth - 1:
                downsample = nn.MaxPool3d(kernel_size=2, stride=2)
                self.downsample_layers.append(downsample)
            
            # 更新通道数
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)  # 限制最大通道数
        
        logger.info(f"Initialized SiameseUNetEncoder with depth={depth}, base_channels={base_channels}")
    
    def _make_encoder_block(
        self, in_channels: int, out_channels: int, dropout_rate: float, use_batch_norm: bool
    ) -> nn.Module:
        """构建编码器块"""
        
        layers = []
        
        # 第一个卷积层
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))
        
        # 第二个卷积层
        layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout3d(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量 [B, C, H, W, D]
            
        Returns:
            features: 最深层的特征图
            skip_connections: 跳跃连接的特征图列表
        """
        skip_connections = []
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            skip_connections.append(x)
            
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        
        return x, skip_connections

class DifferenceFeatureExtractor(nn.Module):
    """差分特征提取器 - 提取时相差异特征"""
    
    def __init__(
        self,
        feature_channels: int,
        difference_method: str = "concat",  # concat, subtract, multiply, attention
        attention_heads: int = 8,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.difference_method = difference_method
        self.attention_heads = attention_heads
        
        if difference_method == "concat":
            # 拼接后降维
            self.diff_conv = nn.Conv3d(feature_channels * 2, feature_channels, kernel_size=1)
        elif difference_method == "subtract":
            # 直接相减，无需额外层
            self.diff_conv = None
        elif difference_method == "multiply":
            # 逐元素相乘后降维
            self.diff_conv = nn.Conv3d(feature_channels, feature_channels, kernel_size=1)
        elif difference_method == "attention":
            # 注意力机制
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_channels,
                num_heads=attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_channels)
        else:
            raise ValueError(f"Unknown difference method: {difference_method}")
        
        logger.info(f"Initialized DifferenceFeatureExtractor with method={difference_method}")
    
    def forward(self, features_t0: Tensor, features_t1: Tensor) -> Tensor:
        """提取差分特征"""
        
        if self.difference_method == "concat":
            # 拼接两个时相的特征
            concat_features = torch.cat([features_t0, features_t1], dim=1)
            diff_features = self.diff_conv(concat_features)
            
        elif self.difference_method == "subtract":
            # 直接相减
            diff_features = features_t1 - features_t0
            
        elif self.difference_method == "multiply":
            # 逐元素相乘
            elementwise_product = features_t1 * features_t0
            diff_features = self.diff_conv(elementwise_product)
            
        elif self.difference_method == "attention":
            # 注意力机制
            B, C, H, W, D = features_t0.shape
            
            # 重塑为序列格式
            features_t0_flat = features_t0.view(B, C, -1).transpose(1, 2)  # [B, H*W*D, C]
            features_t1_flat = features_t1.view(B, C, -1).transpose(1, 2)
            
            # 应用注意力
            attended_t1, _ = self.attention(features_t1_flat, features_t0_flat, features_t0_flat)
            attended_t1 = self.norm(attended_t1 + features_t1_flat)
            
            # 重塑回原始形状
            diff_features = attended_t1.transpose(1, 2).view(B, C, H, W, D)
        
        return diff_features

class TextConditionedDecoder(nn.Module):
    """文本条件解码器 - 支持文本指令的条件分割"""
    
    def __init__(
        self,
        feature_channels: int,
        text_embedding_dim: int,
        output_channels: int = 1,
        depth: int = 4,
        fusion_method: str = "film",  # film, cross_attention, concat
        num_text_tokens: int = 77,  # CLIP文本编码器默认长度
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.text_embedding_dim = text_embedding_dim
        self.output_channels = output_channels
        self.depth = depth
        self.fusion_method = fusion_method
        self.num_text_tokens = num_text_tokens
        
        # 文本特征投影
        self.text_projection = nn.Linear(text_embedding_dim, feature_channels)
        
        # 构建解码器层
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.text_fusion_layers = nn.ModuleList()
        
        # 从深层到浅层
        in_channels = feature_channels
        out_channels = feature_channels // 2
        
        for i in range(depth):
            # 上采样层
            if i > 0:
                upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
                self.upsample_layers.append(upsample)
            
            # 文本融合层
            if fusion_method == "film":
                text_fusion = FiLM(feature_channels, feature_channels)
            elif fusion_method == "cross_attention":
                text_fusion = CrossAttentionFusion(feature_channels, text_embedding_dim)
            elif fusion_method == "concat":
                text_fusion = ConcatFusion(feature_channels, text_embedding_dim)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            self.text_fusion_layers.append(text_fusion)
            
            # 解码块
            decoder_block = self._make_decoder_block(
                in_channels * 2, out_channels, dropout_rate  # *2 for skip connections
            )
            self.decoder_layers.append(decoder_block)
            
            # 更新通道数
            in_channels = out_channels
            out_channels = max(out_channels // 2, 32)  # 限制最小通道数
        
        # 输出层
        self.output_conv = nn.Conv3d(in_channels, output_channels, kernel_size=1)
        
        logger.info(f"Initialized TextConditionedDecoder with fusion_method={fusion_method}")
    
    def _make_decoder_block(self, in_channels: int, out_channels: int, dropout_rate: float) -> nn.Module:
        """构建解码器块"""
        
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        encoder_features: Tensor,
        skip_connections: List[Tensor],
        text_embeddings: Tensor
    ) -> Tensor:
        """前向传播"""
        
        x = encoder_features
        
        # 投影文本特征
        text_features = self.text_projection(text_embeddings)  # [B, text_dim] -> [B, feature_channels]
        
        # 逐层解码
        for i in range(self.depth):
            # 上采样
            if i > 0:
                x = self.upsample_layers[i-1](x)
            
            # 文本条件融合
            x = self.text_fusion_layers[i](x, text_features)
            
            # 跳跃连接
            if i < len(skip_connections):
                skip_feature = skip_connections[-(i+1)]  # 从后往前取跳跃连接
                if x.shape[2:] == skip_feature.shape[2:]:  # 检查空间维度匹配
                    x = torch.cat([x, skip_feature], dim=1)
                else:
                    # 空间维度不匹配时进行插值
                    skip_feature_resized = F.interpolate(
                        skip_feature, size=x.shape[2:], mode='trilinear', align_corners=False
                    )
                    x = torch.cat([x, skip_feature_resized], dim=1)
            
            # 解码块
            x = self.decoder_layers[i](x)
        
        # 输出层
        output = self.output_conv(x)
        
        return output

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) 层"""
    
    def __init__(self, feature_channels: int, text_channels: int):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.text_channels = text_channels
        
        # FiLM参数生成器
        self.film_generator = nn.Sequential(
            nn.Linear(text_channels, feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 2, feature_channels * 2)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: Tensor, text_features: Tensor) -> Tensor:
        """应用FiLM调制"""
        
        B, C, H, W, D = features.shape
        
        # 生成FiLM参数
        film_params = self.film_generator(text_features)  # [B, feature_channels * 2]
        gamma = film_params[:, :C].view(B, C, 1, 1, 1)
        beta = film_params[:, C:].view(B, C, 1, 1, 1)
        
        # 应用FiLM调制
        modulated_features = gamma * features + beta
        
        return modulated_features

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合"""
    
    def __init__(self, feature_channels: int, text_channels: int, num_heads: int = 8):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.text_channels = text_channels
        
        # 特征投影
        self.feature_proj = nn.Conv3d(feature_channels, feature_channels, kernel_size=1)
        self.text_proj = nn.Linear(text_channels, feature_channels)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(feature_channels)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(feature_channels, feature_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_channels * 4, feature_channels)
        )
        
        self.ffn_norm = nn.LayerNorm(feature_channels)
    
    def forward(self, features: Tensor, text_features: Tensor) -> Tensor:
        """应用交叉注意力融合"""
        
        B, C, H, W, D = features.shape
        
        # 投影特征
        features_proj = self.feature_proj(features)  # [B, C, H, W, D]
        text_features_proj = self.text_proj(text_features)  # [B, C]
        
        # 重塑为序列格式
        features_flat = features_proj.view(B, C, -1).transpose(1, 2)  # [B, H*W*D, C]
        text_features_expanded = text_features_proj.unsqueeze(1)  # [B, 1, C]
        
        # 交叉注意力
        attended_features, _ = self.cross_attention(
            features_flat, text_features_expanded, text_features_expanded
        )
        
        # 残差连接和归一化
        attended_features = self.norm(attended_features + features_flat)
        
        # 前馈网络
        ffn_output = self.ffn(attended_features)
        attended_features = self.ffn_norm(ffn_output + attended_features)
        
        # 重塑回原始形状
        output = attended_features.transpose(1, 2).view(B, C, H, W, D)
        
        return output

class ConcatFusion(nn.Module):
    """拼接融合"""
    
    def __init__(self, feature_channels: int, text_channels: int):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.text_channels = text_channels
        
        # 融合层
        self.fusion_conv = nn.Conv3d(feature_channels + text_channels, feature_channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(feature_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, features: Tensor, text_features: Tensor) -> Tensor:
        """应用拼接融合"""
        
        B, C, H, W, D = features.shape
        
        # 扩展文本特征到空间维度
        text_features_expanded = text_features.view(B, self.text_channels, 1, 1, 1)
        text_features_expanded = text_features_expanded.expand(-1, -1, H, W, D)
        
        # 拼接
        concat_features = torch.cat([features, text_features_expanded], dim=1)
        
        # 融合
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.norm(fused_features)
        fused_features = self.activation(fused_features)
        
        return fused_features

class SiameseUNetForLongitudinalSegmentation(nn.Module):
    """用于纵向推理分割的完整Siamese UNet模型"""
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        text_embedding_dim: int = 768,  # CLIP文本编码器维度
        difference_method: str = "concat",
        fusion_method: str = "film",
        dropout_rate: float = 0.1,
        use_deep_supervision: bool = True,
        num_aux_outputs: int = 3
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.depth = depth
        self.text_embedding_dim = text_embedding_dim
        self.difference_method = difference_method
        self.fusion_method = fusion_method
        self.use_deep_supervision = use_deep_supervision
        self.num_aux_outputs = num_aux_outputs
        
        # Siamese编码器
        self.siamese_encoder = SiameseUNetEncoder(
            input_channels=input_channels,
            base_channels=base_channels,
            depth=depth,
            dropout_rate=dropout_rate
        )
        
        # 差分特征提取器
        self.difference_extractor = DifferenceFeatureExtractor(
            feature_channels=base_channels * (2 ** (depth - 1)),
            difference_method=difference_method
        )
        
        # 文本条件解码器
        self.text_conditioned_decoder = TextConditionedDecoder(
            feature_channels=base_channels * (2 ** (depth - 1)),
            text_embedding_dim=text_embedding_dim,
            output_channels=output_channels,
            depth=depth,
            fusion_method=fusion_method,
            dropout_rate=dropout_rate
        )
        
        # 深度监督（可选）
        if use_deep_supervision:
            self.aux_outputs = nn.ModuleList()
            current_channels = base_channels * (2 ** (depth - 1))
            
            for i in range(num_aux_outputs):
                aux_conv = nn.Conv3d(current_channels, output_channels, kernel_size=1)
                self.aux_outputs.append(aux_conv)
                current_channels = max(current_channels // 2, 32)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Initialized SiameseUNetForLongitudinalSegmentation with difference_method={difference_method}, fusion_method={fusion_method}")
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        ct_t0: Tensor,
        ct_t1: Tensor,
        text_embeddings: Tensor,
        return_features: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """前向传播
        
        Args:
            ct_t0: 时相0的CT图像 [B, 1, H, W, D]
            ct_t1: 时相1的CT图像 [B, 1, H, W, D]
            text_embeddings: 文本指令嵌入 [B, text_dim]
            return_features: 是否返回中间特征
            
        Returns:
            segmentation_output: 分割输出 [B, 1, H, W, D]
            features_dict: 中间特征字典（如果return_features=True）
        """
        
        # 编码两个时相
        features_t0, skip_connections_t0 = self.siamese_encoder(ct_t0)
        features_t1, skip_connections_t1 = self.siamese_encoder(ct_t1)
        
        # 提取差分特征
        diff_features = self.difference_extractor(features_t0, features_t1)
        
        # 合并跳跃连接（简单平均）
        merged_skip_connections = []
        for skip_t0, skip_t1 in zip(skip_connections_t0, skip_connections_t1):
            merged_skip = (skip_t0 + skip_t1) / 2.0
            merged_skip_connections.append(merged_skip)
        
        # 文本条件解码
        segmentation_output = self.text_conditioned_decoder(
            diff_features, merged_skip_connections, text_embeddings
        )
        
        # 深度监督输出（可选）
        if self.use_deep_supervision and self.training:
            aux_outputs = []
            current_features = diff_features
            
            for i, aux_conv in enumerate(self.aux_outputs):
                if i < len(merged_skip_connections):
                    # 使用不同层次的特征
                    current_features = merged_skip_connections[-(i+1)]
                
                aux_output = aux_conv(current_features)
                aux_outputs.append(aux_output)
            
            if return_features:
                features_dict = {
                    "features_t0": features_t0,
                    "features_t1": features_t1,
                    "diff_features": diff_features,
                    "skip_connections": merged_skip_connections,
                    "aux_outputs": aux_outputs
                }
                return segmentation_output, features_dict
            else:
                return segmentation_output, aux_outputs
        
        if return_features:
            features_dict = {
                "features_t0": features_t0,
                "features_t1": features_t1,
                "diff_features": diff_features,
                "skip_connections": merged_skip_connections
            }
            return segmentation_output, features_dict
        
        return segmentation_output
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "SiameseUNetForLongitudinalSegmentation",
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "text_embedding_dim": self.text_embedding_dim,
            "difference_method": self.difference_method,
            "fusion_method": self.fusion_method,
            "use_deep_supervision": self.use_deep_supervision,
            "num_aux_outputs": self.num_aux_outputs,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设float32
        }

# 便捷函数
def create_siamese_unet_for_longitudinal_segmentation(
    input_channels: int = 1,
    output_channels: int = 1,
    base_channels: int = 64,
    depth: int = 4,
    text_embedding_dim: int = 768,
    difference_method: str = "concat",
    fusion_method: str = "film",
    **kwargs
) -> SiameseUNetForLongitudinalSegmentation:
    """创建Siamese UNet模型用于纵向推理分割"""
    
    model = SiameseUNetForLongitudinalSegmentation(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        depth=depth,
        text_embedding_dim=text_embedding_dim,
        difference_method=difference_method,
        fusion_method=fusion_method,
        **kwargs
    )
    
    return model

if __name__ == "__main__":
    # 示例用法和测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model = create_siamese_unet_for_longitudinal_segmentation(
        input_channels=1,
        output_channels=1,
        base_channels=32,
        depth=3,
        text_embedding_dim=256,
        difference_method="concat",
        fusion_method="film"
    )
    
    # 打印模型信息
    model_info = model.get_model_info()
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 1
    input_size = (64, 64, 32)
    text_dim = 256
    
    ct_t0 = torch.randn(batch_size, 1, *input_size)
    ct_t1 = torch.randn(batch_size, 1, *input_size)
    text_embeddings = torch.randn(batch_size, text_dim)
    
    print(f"\nTesting forward pass with input shapes:")
    print(f"  ct_t0: {ct_t0.shape}")
    print(f"  ct_t1: {ct_t1.shape}")
    print(f"  text_embeddings: {text_embeddings.shape}")
    
    with torch.no_grad():
        output, features = model(ct_t0, ct_t1, text_embeddings, return_features=True)
        print(f"  output shape: {output.shape}")
        print(f"  features keys: {list(features.keys())}")
    
    print("\nModel test completed successfully!")