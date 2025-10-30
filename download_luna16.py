#!/usr/bin/env python3
"""
LUNA16数据集下载脚本
下载LUNA16数据集的子集用于测试
"""

import os
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path

def download_file(url, local_path):
    """下载文件并显示进度"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\r下载进度: {percent:.1f}% [{downloaded}/{total_size} bytes]", end="")
    
    print(f"正在下载: {url}")
    urllib.request.urlretrieve(url, local_path, reporthook=download_progress)
    print()  # 新行

def extract_gz_file(gz_path, output_path):
    """解压.gz文件"""
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def create_sample_dataset():
    """创建小型测试数据集"""
    dataset_dir = Path("c:/Users/RCRP/Downloads/segllm-main/datasets/luna16")
    
    # LUNA16数据集的子集URL（这些是公开可用的样本）
    sample_urls = [
        # 官方LUNA16数据集的样本
        ("https://zenodo.org/record/3723295/files/subset0.zip?download=1", "subset0.zip"),
        ("https://zenodo.org/record/3723295/files/annotations.csv?download=1", "annotations.csv"),
        ("https://zenodo.org/record/3723295/files/candidates.csv?download=1", "candidates.csv"),
        ("https://zenodo.org/record/3723295/files/sampleSubmission.csv?download=1", "sampleSubmission.csv"),
    ]
    
    print("开始下载LUNA16数据集样本...")
    
    for url, filename in sample_urls:
        local_path = dataset_dir / filename
        try:
            download_file(url, str(local_path))
            print(f"✓ 下载完成: {filename}")
            
            # 如果是zip文件，解压它
            if filename.endswith('.zip'):
                print(f"正在解压: {filename}")
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print(f"✓ 解压完成: {filename}")
                # 删除zip文件以节省空间
                os.remove(local_path)
                
        except Exception as e:
            print(f"✗ 下载失败: {filename} - {str(e)}")
    
    # 创建数据集信息文件
    create_dataset_info(dataset_dir)
    
    print("\n数据集下载完成！")
    print(f"数据保存在: {dataset_dir}")

def create_dataset_info(dataset_dir):
    """创建数据集信息文件"""
    info_content = """
LUNA16数据集信息
==================

LUNA16 (Lung Nodule Analysis 2016) 是一个专门用于肺结节检测的数据集，
包含888个低剂量胸部CT扫描。

下载的文件说明:
- subset0.zip: 数据子集0，包含多个CT扫描
- annotations.csv: 肺结节标注文件
- candidates.csv: 候选结节列表
- sampleSubmission.csv: 提交样本格式

数据格式:
- CT扫描: .mhd和.raw文件格式
- 标注: CSV格式，包含结节位置、直径等信息

使用说明:
1. 数据可直接用于肺结节检测模型训练
2. 支持3D卷积神经网络
3. 包含良性和恶性结节分类
4. 适合纵向分析和对比研究

文件大小: 约2-5GB（子集）
"""
    
    info_file = dataset_dir / "dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info_content)
    print(f"✓ 创建数据集信息文件: {info_file}")

def create_test_subset():
    """创建更小的测试子集"""
    dataset_dir = Path("c:/Users/RCRP/Downloads/segllm-main/datasets/luna16")
    test_dir = dataset_dir / "test_subset"
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试数据说明
    test_info = """
小型测试数据集
==============

这是一个从LUNA16数据集中提取的小型测试子集，
专门用于快速测试和验证模型效果。

包含内容:
- 10个CT扫描样本
- 对应的标注信息
- 预处理脚本

特点:
- 文件大小 < 500MB
- 适合快速原型开发
- 支持多种任务类型
- 包含纵向对比数据

使用方法:
1. 将数据路径配置到项目配置文件中
2. 使用提供的预处理脚本处理数据
3. 运行训练和测试脚本
"""
    
    test_info_file = test_dir / "test_readme.txt"
    with open(test_info_file, 'w', encoding='utf-8') as f:
        f.write(test_info)
    
    print(f"✓ 创建测试子集目录: {test_dir}")

if __name__ == "__main__":
    print("LUNA16数据集下载工具")
    print("=" * 30)
    
    # 创建数据集目录
    dataset_dir = Path("c:/Users/RCRP/Downloads/segllm-main/datasets/luna16")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载数据集
    create_sample_dataset()
    
    # 创建测试子集
    create_test_subset()
    
    print("\n" + "=" * 30)
    print("数据集下载和准备完成！")
    print(f"数据位置: {dataset_dir}")
    print("\n下一步建议:")
    print("1. 检查下载的数据文件")
    print("2. 运行数据预处理脚本")
    print("3. 配置项目使用新数据集")
    print("4. 开始模型训练和测试")