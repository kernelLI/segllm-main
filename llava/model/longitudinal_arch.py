"""
纵向推理模型架构扩展
支持双时相图像输入和差分特征处理

输入:
- images_dict: 纵向图像对字典，包含"t0"和"t1"键，值为图像张量[batch_size, channels, height, width]
- input_ids: 输入token ID序列[batch_size, seq_len]
- attention_mask: 注意力掩码[batch_size, seq_len]
- labels: 标签序列（可选）[batch_size, seq_len]
- config: 模型配置对象，包含enable_longitudinal、mm_hidden_size等参数

输出:
- image_features: 融合后的图像特征[batch_size, seq_len, hidden_size]
- change_features: 变化特征（通过change_projector降维后）[batch_size, seq_len, hidden_size]
- change_scores: 变化检测分数（可选）[batch_size, seq_len, 1]
- new_input_embeds: 融合后的输入嵌入[batch_size, new_seq_len, embed_dim]
- new_labels: 更新后的标签（如果提供）[batch_size, new_seq_len]

功能:
- 扩展LlavaMetaModel支持纵向图像对处理
- 实现5种特征融合方式：原始特征拼接、差分特征、相加特征、相乘特征
- 提供变化检测与门控机制，增强变化区域特征
- 支持单图像和纵向图像对两种模式
- 保持与原有SegLLM框架的兼容性
- 实现变化投影器降维和特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import (
    DEFAULT_IMAGE_T0_TOKEN, DEFAULT_IMAGE_T1_TOKEN, DEFAULT_CHANGE_TOKEN,
    IMAGE_TOKEN_INDEX, IGNORE_INDEX
)
import logging

logger = logging.getLogger(__name__)

class LongitudinalMetaModel(LlavaMetaModel):
    """扩展LlavaMetaModel支持纵向推理"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # 添加差分特征处理器 - 支持5种特征融合
        if hasattr(config, 'enable_longitudinal') and config.enable_longitudinal:
            self.change_projector = nn.Linear(
                config.mm_hidden_size * 5,  # 拼接5种特征：t0+t1+diff+add+mul
                config.mm_hidden_size
            )
            
            # 可选：添加变化检测头
            if getattr(config, 'use_change_detection', False):
                self.change_detector = nn.Sequential(
                    nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(config.mm_hidden_size // 2, 1),
                    nn.Sigmoid()
                )
    
    def encode_longitudinal_images(self, images_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        编码纵向图像对 - 支持多种特征融合方式
        
        Args:
            images_dict: 包含t0和t1图像的字典
                {
                    "t0": [batch_size, channels, height, width],
                    "t1": [batch_size, channels, height, width]
                }
        
        Returns:
            融合后的特征张量 [batch_size, seq_len, hidden_size]
        """
        # 分别编码两个时相的图像
        vision_tower = self.get_vision_tower()
        mm_projector = self.mm_projector
        
        # 编码t0图像
        image_features_t0 = vision_tower(images_dict["t0"])
        image_features_t0 = mm_projector(image_features_t0)
        
        # 编码t1图像  
        image_features_t1 = vision_tower(images_dict["t1"])
        image_features_t1 = mm_projector(image_features_t1)
        
        # 确保特征维度一致
        assert image_features_t0.shape == image_features_t1.shape, \
            f"特征维度不匹配: t0 {image_features_t0.shape} vs t1 {image_features_t1.shape}"
        
        # 多特征融合：拼接 + 差分 + 相加 + 相乘
        diff_features = image_features_t1 - image_features_t0
        add_features = image_features_t1 + image_features_t0  
        mul_features = image_features_t1 * image_features_t0
        
        # 特征拼接 [batch_size, seq_len, hidden_size * 5] - 让网络学习最优融合
        combined_features = torch.cat([
            image_features_t0, image_features_t1,
            diff_features, add_features, mul_features
        ], dim=-1)
        
        # 通过变化投影器降维
        if hasattr(self, 'change_projector'):
            change_features = self.change_projector(combined_features)
        else:
            # 如果没有变化投影器，使用平均池化
            change_features = (image_features_t0 + image_features_t1) / 2
        
        # 可选：添加变化检测与门控机制
        if hasattr(self, 'change_detector'):
            change_scores = self.change_detector(change_features)
            logger.debug(f"Change detection scores: {change_scores.mean().item():.4f}")
            # 门控机制：增强变化区域特征
            change_features = change_features * (1 + change_scores)
        
        return change_features
    
    def encode_single_image(self, images):
        """保持原有的单图像编码功能"""
        return super().encode_images(images)

class LongitudinalMetaForCausalLM(LlavaMetaForCausalLM):
    """扩展纵向推理功能的元类"""
    
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images,
        novision=False
    ):
        """
        准备多模态输入标签，支持纵向图像对
        
        Args:
            images: 可以是单图像或纵向图像对
                - 单图像: torch.Tensor 或 dict 包含 "image" 键
                - 纵向对: dict 包含 "t0" 和 "t1" 键
        """
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1 or novision:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return None, attention_mask, past_key_values, self.get_model().embed_tokens(input_ids), labels
        
        # 判断是否为纵向图像对
        is_longitudinal = isinstance(images, dict) and "t0" in images and "t1" in images
        
        if is_longitudinal:
            # 纵向推理模式
            image_features = self.get_model().encode_longitudinal_images(images)
        else:
            # 普通单图像模式
            image_features = self.encode_images(images)
        
        # 使用父类方法处理特征嵌入
        return self._process_multimodal_embeddings(
            input_ids, attention_mask, past_key_values, labels, image_features
        )
    
    def _process_multimodal_embeddings(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        image_features
    ):
        """处理多模态特征嵌入"""
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # 检查是否有图像token
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            if image_token_indices.numel() == 0:
                # 纯文本模式
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue
            
            # 处理包含图像token的序列
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
            
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                if len(cur_image_features.shape) == 1:
                    cur_image_features = cur_image_features[None,]
                
                image_token_start = image_token_indices[0]
                
                # 添加图像token之前的文本嵌入
                cur_new_input_embeds.append(
                    self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                )
                
                # 添加图像特征
                cur_new_input_embeds.append(cur_image_features)
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=labels.device,
                            dtype=labels.dtype
                        )
                    )
                    cur_labels = cur_labels[image_token_start + 1:]
                
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            
            # 添加剩余文本
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            
            # 合并嵌入
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        
        # 处理长度不一致的情况
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            new_input_embeds_padded = []
            new_labels_padded = [] if labels is not None else None
            
            for i, embed in enumerate(new_input_embeds):
                if embed.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - embed.shape[0], embed.shape[1],
                        dtype=embed.dtype, device=embed.device
                    )
                    embed = torch.cat([embed, padding], dim=0)
                new_input_embeds_padded.append(embed)
                
                if labels is not None:
                    label = new_labels[i]
                    if label.shape[0] < max_len:
                        padding = torch.full(
                            (max_len - label.shape[0],),
                            IGNORE_INDEX,
                            dtype=label.dtype, device=label.device
                        )
                        label = torch.cat([label, padding], dim=0)
                    new_labels_padded.append(label)
            
            new_input_embeds = new_input_embeds_padded
            if labels is not None:
                new_labels = new_labels_padded
        
        # 堆叠批次
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)
        
        return None, attention_mask, past_key_values, new_input_embeds, new_labels