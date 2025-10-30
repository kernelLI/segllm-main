"""
文本条件调制模块 - 用于LIDC-IDRI纵向推理分割
支持多种文本编码器和条件调制方法，集成到SegLLM架构中
"""

import os
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from transformers import (
    AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer,
    BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
)

logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """文本编码器基类"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 77):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # 根据模型名称选择对应的编码器
        if "clip" in model_name.lower():
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)
            self.embedding_dim = self.text_model.config.hidden_size
        elif "bert" in model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.text_model = BertModel.from_pretrained(model_name)
            self.embedding_dim = self.text_model.config.hidden_size
        elif "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.text_model = RobertaModel.from_pretrained(model_name)
            self.embedding_dim = self.text_model.config.hidden_size
        else:
            # 默认使用AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModel.from_pretrained(model_name)
            self.embedding_dim = self.text_model.config.hidden_size
        
        # 冻结预训练权重（可选）
        self.freeze_weights = False
        if self.freeze_weights:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        logger.info(f"Initialized TextEncoder with {model_name}, embedding_dim={self.embedding_dim}")
    
    def forward(self, texts: List[str]) -> Dict[str, Tensor]:
        """编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            dict: 包含文本嵌入和其他信息的字典
        """
        
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到相同设备
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 编码
        with torch.set_grad_enabled(not self.freeze_weights):
            outputs = self.text_model(**inputs)
            
            # 提取文本嵌入
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                text_embeddings = outputs.pooler_output
            else:
                # 使用[CLS]token或平均池化
                last_hidden_state = outputs.last_hidden_state
                if self.tokenizer.cls_token_id is not None:
                    cls_token_idx = inputs['input_ids'] == self.tokenizer.cls_token_id
                    text_embeddings = last_hidden_state[cls_token_idx].view(len(texts), -1)
                else:
                    # 平均池化
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    text_embeddings = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        return {
            "text_embeddings": text_embeddings,
            "attention_mask": inputs['attention_mask'],
            "input_ids": inputs['input_ids']
        }
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> Tensor:
        """批量编码文本"""
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.forward(batch_texts)
            all_embeddings.append(batch_results["text_embeddings"])
        
        return torch.cat(all_embeddings, dim=0)

class TextFeatureExtractor(nn.Module):
    """文本特征提取器 - 从医学文本中提取关键特征"""
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        feature_types: List[str] = ["volume", "density", "morphology", "change_type"],
        use_regex_patterns: bool = True
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.feature_types = feature_types
        self.use_regex_patterns = use_regex_patterns
        
        # 定义特征提取的正则表达式模式
        self.patterns = {
            "volume": {
                "increase": r"增加|增大|变大|增长|上升|≥|>=|>|超过",
                "decrease": r"减少|减小|变小|下降|降低|<|<=|≤|少于",
                "stable": r"稳定|不变|无变化|保持",
                "percentage": r"(\d+(?:\.\d+)?)\s*%",
                "absolute": r"(\d+(?:\.\d+)?)\s*(?:mm|cm|ml|cc)"
            },
            "density": {
                "increase": r"密度增加|变实|实性化|HU增加|CT值增加",
                "decrease": r"密度减少|变磨玻璃|磨玻璃化|HU减少|CT值减少",
                "ground_glass": r"磨玻璃|GGO|GGN|ground.glass",
                "solid": r"实性|solid|dense",
                "mixed": r"混合|部分实性|part.solid",
                "hu_value": r"(\d+(?:\.\d+)?)\s*HU",
                "ct_value": r"(\d+(?:\.\d+)?)\s*CT"
            },
            "morphology": {
                "smooth": r"光滑|smooth|规则|regular",
                "lobulated": r"分叶|lobulated|不规则|irregular",
                "spiculated": r"毛刺|spiculated|毛刺征",
                "well_defined": r"边界清晰|well.defined|清楚|清晰",
                "ill_defined": r"边界模糊|ill.defined|模糊|不清楚",
                "round": r"圆形|round|球形",
                "oval": r"椭圆形|oval|卵圆形",
                "irregular": r"不规则|irregular|形态不规则"
            },
            "change_type": {
                "new": r"新出现|new|新发|新增|出现",
                "disappear": r"消失|disappear|消散|缩小至无",
                "grow": r"生长|growth|增大|progression",
                "shrink": r"缩小|shrink|regression|好转",
                "stable": r"稳定|stable|无变化|unchanged"
            }
        }
        
        # 特征映射层
        self.feature_mappers = nn.ModuleDict()
        for feature_type in feature_types:
            self.feature_mappers[feature_type] = nn.Sequential(
                nn.Linear(text_encoder.embedding_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32)  # 每个特征类型输出32维向量
            )
        
        logger.info(f"Initialized TextFeatureExtractor with feature_types={feature_types}")
    
    def extract_features_from_text(self, texts: List[str]) -> Dict[str, Tensor]:
        """从文本中提取结构化特征"""
        
        extracted_features = {}
        
        for i, text in enumerate(texts):
            text_features = {}
            
            # 使用正则表达式提取特征
            if self.use_regex_patterns:
                for feature_type, patterns in self.patterns.items():
                    feature_values = {}
                    
                    for pattern_name, pattern in patterns.items():
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            if pattern_name in ["percentage", "absolute", "hu_value", "ct_value"]:
                                # 数值特征
                                try:
                                    feature_values[pattern_name] = float(matches[0])
                                except (ValueError, IndexError):
                                    feature_values[pattern_name] = 0.0
                            else:
                                # 布尔特征
                                feature_values[pattern_name] = 1.0
                        else:
                            if pattern_name in ["percentage", "absolute", "hu_value", "ct_value"]:
                                feature_values[pattern_name] = 0.0
                            else:
                                feature_values[pattern_name] = 0.0
                    
                    text_features[feature_type] = feature_values
            
            extracted_features[i] = text_features
        
        return extracted_features
    
    def forward(self, texts: List[str]) -> Dict[str, Tensor]:
        """前向传播"""
        
        # 编码文本
        text_results = self.text_encoder(texts)
        text_embeddings = text_results["text_embeddings"]
        
        # 提取结构化特征
        structured_features = self.extract_features_from_text(texts)
        
        # 映射特征
        mapped_features = {}
        for feature_type in self.feature_types:
            if feature_type in self.feature_mappers:
                mapped_features[feature_type] = self.feature_mappers[feature_type](text_embeddings)
        
        return {
            "text_embeddings": text_embeddings,
            "structured_features": structured_features,
            "mapped_features": mapped_features,
            "attention_mask": text_results["attention_mask"]
        }

class TextConditionedModulation(nn.Module):
    """文本条件调制模块 - 将文本特征注入到图像特征中"""
    
    def __init__(
        self,
        image_channels: int,
        text_channels: int,
        modulation_type: str = "film",  # film, attention, gate, concat
        num_layers: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.text_channels = text_channels
        self.modulation_type = modulation_type
        self.num_layers = num_layers
        
        # 构建调制层
        self.modulation_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if modulation_type == "film":
                layer = FiLMModulation(image_channels, text_channels)
            elif modulation_type == "attention":
                layer = AttentionModulation(image_channels, text_channels)
            elif modulation_type == "gate":
                layer = GatedModulation(image_channels, text_channels)
            elif modulation_type == "concat":
                layer = ConcatModulation(image_channels, text_channels)
            else:
                raise ValueError(f"Unknown modulation type: {modulation_type}")
            
            self.modulation_layers.append(layer)
        
        logger.info(f"Initialized TextConditionedModulation with type={modulation_type}, layers={num_layers}")
    
    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """应用文本条件调制"""
        
        x = image_features
        
        for modulation_layer in self.modulation_layers:
            x = modulation_layer(x, text_features)
        
        return x

class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation (FiLM) 调制"""
    
    def __init__(self, image_channels: int, text_channels: int):
        super().__init__()
        
        self.image_channels = image_channels
        self.text_channels = text_channels
        
        # FiLM参数生成器
        self.film_generator = nn.Sequential(
            nn.Linear(text_channels, image_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(image_channels * 4, image_channels * 2)
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
    
    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """应用FiLM调制"""
        
        B, C, H, W, D = image_features.shape
        
        # 生成FiLM参数
        film_params = self.film_generator(text_features)  # [B, image_channels * 2]
        gamma = film_params[:, :C].view(B, C, 1, 1, 1)
        beta = film_params[:, C:].view(B, C, 1, 1, 1)
        
        # 应用FiLM调制
        modulated_features = gamma * image_features + beta
        
        return modulated_features

class AttentionModulation(nn.Module):
    """注意力调制"""
    
    def __init__(self, image_channels: int, text_channels: int, num_heads: int = 8):
        super().__init__()
        
        self.image_channels = image_channels
        self.text_channels = text_channels
        
        # 特征投影
        self.image_proj = nn.Conv3d(image_channels, image_channels, kernel_size=1)
        self.text_proj = nn.Linear(text_channels, image_channels)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(image_channels)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(image_channels, image_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(image_channels * 4, image_channels)
        )
        
        self.ffn_norm = nn.LayerNorm(image_channels)
    
    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """应用注意力调制"""
        
        B, C, H, W, D = image_features.shape
        
        # 投影特征
        image_proj = self.image_proj(image_features)  # [B, C, H, W, D]
        text_proj = self.text_proj(text_features)  # [B, C]
        
        # 重塑为序列格式
        image_flat = image_proj.view(B, C, -1).transpose(1, 2)  # [B, H*W*D, C]
        text_expanded = text_proj.unsqueeze(1)  # [B, 1, C]
        
        # 交叉注意力
        attended_features, _ = self.cross_attention(
            image_flat, text_expanded, text_expanded
        )
        
        # 残差连接和归一化
        attended_features = self.norm(attended_features + image_flat)
        
        # 前馈网络
        ffn_output = self.ffn(attended_features)
        attended_features = self.ffn_norm(ffn_output + attended_features)
        
        # 重塑回原始形状
        output = attended_features.transpose(1, 2).view(B, C, H, W, D)
        
        return output

class GatedModulation(nn.Module):
    """门控调制"""
    
    def __init__(self, image_channels: int, text_channels: int):
        super().__init__()
        
        self.image_channels = image_channels
        self.text_channels = text_channels
        
        # 门控生成器
        self.gate_generator = nn.Sequential(
            nn.Linear(text_channels, image_channels),
            nn.Sigmoid()
        )
        
        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Conv3d(image_channels, image_channels, kernel_size=1),
            nn.BatchNorm3d(image_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """应用门控调制"""
        
        B, C, H, W, D = image_features.shape
        
        # 生成门控信号
        gate = self.gate_generator(text_features)  # [B, image_channels]
        gate = gate.view(B, C, 1, 1, 1)
        
        # 变换特征
        transformed_features = self.feature_transform(image_features)
        
        # 应用门控
        gated_features = gate * transformed_features + (1 - gate) * image_features
        
        return gated_features

class ConcatModulation(nn.Module):
    """拼接调制"""
    
    def __init__(self, image_channels: int, text_channels: int):
        super().__init__()
        
        self.image_channels = image_channels
        self.text_channels = text_channels
        
        # 融合层
        self.fusion_conv = nn.Conv3d(image_channels + text_channels, image_channels, kernel_size=1)
        self.norm = nn.BatchNorm3d(image_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, image_features: Tensor, text_features: Tensor) -> Tensor:
        """应用拼接调制"""
        
        B, C, H, W, D = image_features.shape
        
        # 扩展文本特征到空间维度
        text_expanded = text_features.view(B, self.text_channels, 1, 1, 1)
        text_expanded = text_expanded.expand(-1, -1, H, W, D)
        
        # 拼接
        concat_features = torch.cat([image_features, text_expanded], dim=1)
        
        # 融合
        fused_features = self.fusion_conv(concat_features)
        fused_features = self.norm(fused_features)
        fused_features = self.activation(fused_features)
        
        return fused_features

class LongitudinalTextConditioning(nn.Module):
    """纵向推理的文本条件调制模块"""
    
    def __init__(
        self,
        text_encoder_name: str = "bert-base-uncased",
        image_channels: int = 64,
        modulation_type: str = "film",
        feature_types: List[str] = ["volume", "density", "morphology", "change_type"],
        max_text_length: int = 77,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.text_encoder_name = text_encoder_name
        self.image_channels = image_channels
        self.modulation_type = modulation_type
        self.feature_types = feature_types
        self.max_text_length = max_text_length
        
        # 文本编码器
        self.text_encoder = TextEncoder(text_encoder_name, max_text_length)
        
        # 文本特征提取器
        self.text_feature_extractor = TextFeatureExtractor(
            self.text_encoder, feature_types
        )
        
        # 文本条件调制
        self.text_conditioning = TextConditionedModulation(
            image_channels=image_channels,
            text_channels=self.text_encoder.embedding_dim,
            modulation_type=modulation_type,
            dropout_rate=dropout_rate
        )
        
        logger.info(f"Initialized LongitudinalTextConditioning with {text_encoder_name}, modulation={modulation_type}")
    
    def forward(
        self,
        image_features: Tensor,
        texts: List[str],
        return_text_features: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """前向传播"""
        
        # 提取文本特征
        text_results = self.text_feature_extractor(texts)
        text_embeddings = text_results["text_embeddings"]
        
        # 应用文本条件调制
        conditioned_features = self.text_conditioning(image_features, text_embeddings)
        
        if return_text_features:
            return conditioned_features, {
                "text_embeddings": text_embeddings,
                "structured_features": text_results["structured_features"],
                "mapped_features": text_results["mapped_features"]
            }
        
        return conditioned_features
    
    def get_text_embedding_dim(self) -> int:
        """获取文本嵌入维度"""
        return self.text_encoder.embedding_dim
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        return {
            "text_encoder_name": self.text_encoder_name,
            "image_channels": self.image_channels,
            "modulation_type": self.modulation_type,
            "feature_types": self.feature_types,
            "text_embedding_dim": self.get_text_embedding_dim()
        }

# 便捷函数
def create_longitudinal_text_conditioning(
    text_encoder_name: str = "bert-base-uncased",
    image_channels: int = 64,
    modulation_type: str = "film",
    feature_types: List[str] = ["volume", "density", "morphology", "change_type"],
    **kwargs
) -> LongitudinalTextConditioning:
    """创建纵向文本条件调制模块"""
    
    return LongitudinalTextConditioning(
        text_encoder_name=text_encoder_name,
        image_channels=image_channels,
        modulation_type=modulation_type,
        feature_types=feature_types,
        **kwargs
    )

if __name__ == "__main__":
    # 示例用法和测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建文本条件调制模块
    text_conditioning = create_longitudinal_text_conditioning(
        text_encoder_name="bert-base-uncased",
        image_channels=64,
        modulation_type="film",
        feature_types=["volume", "density", "morphology", "change_type"]
    )
    
    # 测试文本
    test_texts = [
        "分割所有体积增加超过25%的结节",
        "标出新出现的磨玻璃结节",
        "圈出边界由清晰变模糊的病灶"
    ]
    
    # 测试图像特征
    batch_size = len(test_texts)
    image_channels = 64
    spatial_size = (32, 32, 16)
    image_features = torch.randn(batch_size, image_channels, *spatial_size)
    
    print(f"Testing with:")
    print(f"  Texts: {test_texts}")
    print(f"  Image features shape: {image_features.shape}")
    
    # 前向传播
    with torch.no_grad():
        conditioned_features, text_info = text_conditioning(
            image_features, test_texts, return_text_features=True
        )
        
        print(f"  Conditioned features shape: {conditioned_features.shape}")
        print(f"  Text embeddings shape: {text_info['text_embeddings'].shape}")
        print(f"  Structured features keys: {list(text_info['structured_features'][0].keys())}")
        print(f"  Mapped features keys: {list(text_info['mapped_features'].keys())}")
    
    # 获取模型信息
    feature_info = text_conditioning.get_feature_info()
    print(f"\nFeature info: {feature_info}")
    
    print("\nText conditioning test completed successfully!")