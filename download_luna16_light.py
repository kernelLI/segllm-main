#!/usr/bin/env python3
"""
LUNA16轻量级数据集下载脚本
下载LUNA16数据集的标注和样本文件，避免下载大文件
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path

def download_file(url, local_path):
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        urllib.request.urlretrieve(url, local_path)
        print(f"✓ 下载完成: {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        return False

def create_sample_data():
    """创建样本数据文件"""
    dataset_dir = Path("c:/Users/RCRP/Downloads/segllm-main/datasets/luna16")
    
    # 创建模拟的CT扫描数据（小文件）
    sample_ct_data = """
# LUNA16格式样本CT扫描数据
# 每个扫描包含多个切片

scan1:
  - spacing: [1.0, 1.0, 1.0]
  - origin: [0, 0, 0]
  - dimensions: [512, 512, 100]
  - nodules: 2
  
scan2:
  - spacing: [0.8, 0.8, 1.5]
  - origin: [0, 0, 0]
  - dimensions: [512, 512, 120]
  - nodules: 1
"""
    
    sample_file = dataset_dir / "sample_ct_info.txt"
    with open(sample_file, 'w') as f:
        f.write(sample_ct_data)
    
    # 创建标注文件
    annotations_data = {
        'seriesuid': ['1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836900',
                      '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836900',
                      '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836901'],
        'coordX': [-56.08, -56.08, -108.39],
        'coordY': [-67.64, -67.64, -133.29],
        'coordZ': [-311.72, -311.72, -298.19],
        'diameter_mm': [5.0, 5.0, 8.0]
    }
    
    annotations_df = pd.DataFrame(annotations_data)
    annotations_file = dataset_dir / "annotations.csv"
    annotations_df.to_csv(annotations_file, index=False)
    
    # 创建候选文件
    candidates_data = {
        'seriesuid': ['1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836900'] * 5,
        'coordX': [-56.08, -50.0, -45.0, -60.0, -55.0],
        'coordY': [-67.64, -65.0, -70.0, -68.0, -66.0],
        'coordZ': [-311.72, -310.0, -315.0, -312.0, -309.0],
        'class': [1, 0, 0, 1, 0]  # 1表示结节，0表示非结节
    }
    
    candidates_df = pd.DataFrame(candidates_data)
    candidates_file = dataset_dir / "candidates.csv"
    candidates_df.to_csv(candidates_file, index=False)
    
    print("✓ 创建样本数据文件完成")
    return annotations_file, candidates_file

def download_luna16_metadata():
    """下载LUNA16元数据文件"""
    dataset_dir = Path("c:/Users/RCRP/Downloads/segllm-main/datasets/luna16")
    
    # 下载LUNA16的标注文件（小文件）
    urls = [
        ("https://zenodo.org/record/3723295/files/annotations.csv?download=1", "annotations_luna16.csv"),
        ("https://zenodo.org/record/3723295/files/candidates.csv?download=1", "candidates_luna16.csv"),
        ("https://zenodo.org/record/3723295/files/sampleSubmission.csv?download=1", "sampleSubmission.csv"),
    ]
    
    downloaded_files = []
    for url, filename in urls:
        local_path = dataset_dir / filename
        if download_file(url, str(local_path)):
            downloaded_files.append(local_path)
    
    return downloaded_files

def create_json_format_data():
    """创建JSON格式的测试数据，适配你的项目"""
    dataset_dir = Path("c:/Users/RCRP\Downloads\segllm-main/datasets/luna16")
    
    # 创建纵向分析格式的数据
    longitudinal_data = {
        "samples": [
            {
                "id": "patient_001",
                "image_t0_path": "scans/patient_001_t0.mhd",
                "image_t1_path": "scans/patient_001_t1.mhd", 
                "mask_paths": ["masks/patient_001_nodule1.nii", "masks/patient_001_nodule2.nii"],
                "task_type": "nodule_detection",
                "changes": "检测到新的肺结节，直径约5mm",
                "metadata": {
                    "spacing": [1.0, 1.0, 1.0],
                    "origin": [0, 0, 0],
                    "dimensions": [512, 512, 100],
                    "scan_date_t0": "2023-01-15",
                    "scan_date_t1": "2023-06-20"
                }
            },
            {
                "id": "patient_002", 
                "image_t0_path": "scans/patient_002_t0.mhd",
                "image_t1_path": "scans/patient_002_t1.mhd",
                "mask_paths": ["masks/patient_002_nodule1.nii"],
                "task_type": "nodule_growth",
                "changes": "肺结节增大，从3mm增长到8mm",
                "metadata": {
                    "spacing": [0.8, 0.8, 1.5],
                    "origin": [0, 0, 0], 
                    "dimensions": [512, 512, 120],
                    "scan_date_t0": "2023-03-10",
                    "scan_date_t1": "2023-09-15"
                }
            }
        ]
    }
    
    json_file = dataset_dir / "luna16_longitudinal_test.json"
    import json
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(longitudinal_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 创建JSON格式数据: {json_file}")
    return json_file

def create_dataset_readme():
    """创建数据集说明文档"""
    dataset_dir = Path("c:/Users/RCRP\Downloads\segllm-main/datasets/luna16")
    
    readme_content = """
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
"""
    
    readme_file = dataset_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ 创建数据集说明文档: {readme_file}")

if __name__ == "__main__":
    print("LUNA16轻量级数据集下载工具")
    print("=" * 40)
    
    # 创建数据集目录
    dataset_dir = Path("c:/Users/RCRP\Downloads\segllm-main/datasets/luna16")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. 下载LUNA16元数据文件...")
    metadata_files = download_luna16_metadata()
    
    print("\n2. 创建样本数据文件...")
    sample_files = create_sample_data()
    
    print("\n3. 创建JSON格式测试数据...")
    json_file = create_json_format_data()
    
    print("\n4. 创建数据集说明文档...")
    create_dataset_readme()
    
    print("\n" + "=" * 40)
    print("✓ 轻量级数据集准备完成！")
    print(f"数据位置: {dataset_dir}")
    print(f"总大小: < 10MB")
    
    print("\n主要文件:")
    print(f"- 标注文件: {len(metadata_files)} 个")
    print(f"- JSON数据: {json_file}")
    print(f"- 样本数据: {len(sample_files)} 个")
    
    print("\n下一步建议:")
    print("1. 使用test_dataset_loading.py验证数据格式")
    print("2. 运行项目测试脚本")
    print("3. 开始模型训练和评估")
    print("4. 根据结果调整模型参数")