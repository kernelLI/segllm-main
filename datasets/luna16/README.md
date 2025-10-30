
# LUNA16轻量级测试数据集

## 数据集概述
LUNA16 (Lung Nodule Analysis 2016) 是一个专门用于肺结节检测的医学影像数据集。
这个轻量级版本包含了数据集的元数据和样本，适合快速测试和开发。

## 文件结构
```
luna16/
├── annotations_luna16.csv    # LUNA16官方标注文件
├── candidates_luna16.csv     # LUNA16候选结节列表
├── annotations.csv            # 简化版标注文件
├── candidates.csv             # 简化版候选文件
├── sampleSubmission.csv       # 提交格式样本
├── luna16_longitudinal_test.json  # 纵向分析测试数据
├── sample_ct_info.txt         # CT扫描样本信息
├── test_subset/               # 测试子集目录
└── dataset_info.txt           # 数据集详细信息
```

## 数据特点
- **文件大小**: < 50MB (轻量级版本)
- **扫描数量**: 888个完整CT扫描 (完整版)
- **结节数量**: 1186个标注结节 (完整版)
- **图像格式**: MHD/RAW, DICOM
- **标注格式**: CSV, JSON
- **适合任务**: 肺结节检测、分割、分类

## 快速开始
1. 数据已预处理为JSON格式，可直接用于项目
2. 支持纵向时间序列分析
3. 包含多语言支持（中英文）
4. 提供完整的训练和测试流程

## 使用建议
- 先用小数据集验证模型效果
- 逐步扩展到完整数据集
- 注意GPU内存限制
- 使用数据增强提高性能

## 扩展选项
如需完整数据集，请访问:
- 官方网站: https://luna16.grand-challenge.org/
- 下载完整CT扫描图像
- 文件大小约60GB
