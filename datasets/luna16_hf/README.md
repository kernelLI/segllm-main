# LUNA16 Lung Nodule Detection Dataset

## 数据集概述
LUNA16 (Lung Nodule Analysis 16) 是一个专门用于肺结节检测的医学影像数据集。
本版本为Hugging Face兼容格式，适合快速原型开发和测试。

## 数据集统计
- 样本数量: 10
- 标注结节数量: 11
- 候选检测结果: 107

## 文件结构
```
luna16_hf/
├── samples.json          # CT扫描样本数据
├── annotations.csv       # 结节标注信息
├── candidates.csv        # 候选检测结果
└── dataset_info.json     # 数据集元信息
```

## 使用示例
```python
import json
import pandas as pd

# 加载样本数据
with open('datasets/luna16_hf/samples.json', 'r') as f:
    samples = json.load(f)

# 加载标注数据
annotations = pd.read_csv('datasets/luna16_hf/annotations.csv')

# 加载候选数据
candidates = pd.read_csv('datasets/luna16_hf/candidates.csv')
```

## 数据格式
- **samples.json**: CT扫描的元数据，包括患者信息、扫描参数、结节列表等
- **annotations.csv**: 专业医师标注的结节位置、大小、性质等信息
- **candidates.csv**: 算法生成的候选结节检测结果

## 引用
如果您使用此数据集，请引用原始LUNA16数据集：
```
LUNA16: https://luna16.grand-challenge.org/
```
