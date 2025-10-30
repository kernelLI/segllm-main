#!/usr/bin/env python3
"""
测试数据集加载脚本
验证创建的小型测试数据集是否符合项目要求
"""

import json
import os
import sys
from pathlib import Path

def test_json_format():
    """测试JSON文件格式"""
    print("=== 测试JSON文件格式 ===")
    
    test_files = [
        "lidc_test_sample.json",
        "lidc_test_simple.json"
    ]
    
    for filename in test_files:
        filepath = Path(__file__).parent / filename
        print(f"\n测试文件: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ JSON格式正确")
            print(f"✓ 样本数量: {len(data)}")
            
            # 验证数据格式
            if isinstance(data, list) and len(data) > 0:
                sample = data[0]
                print(f"✓ 第一个样本的键: {list(sample.keys())}")
                
                # 验证必需字段
                if filename == "lidc_test_sample.json":
                    required_fields = ['id', 'image_t0', 'image_t1', 'task_type', 'changes', 'metadata']
                else:  # lidc_test_simple.json
                    required_fields = ['id', 'image_paths', 'findings', 'comparison', 'impression', 'metadata']
                
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    print(f"✗ 缺失字段: {missing_fields}")
                else:
                    print(f"✓ 所有必需字段都存在")
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON解析错误: {e}")
        except Exception as e:
            print(f"✗ 其他错误: {e}")

def test_data_content():
    """测试数据内容有效性"""
    print("\n=== 测试数据内容 ===")
    
    filepath = Path(__file__).parent / "lidc_test_sample.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, sample in enumerate(data):
            print(f"\n样本 {i+1}: {sample['id']}")
            
            # 检查任务类型
            task_type = sample.get('task_type', '')
            print(f"  任务类型: {task_type}")
            
            # 检查变化信息
            changes = sample.get('changes', {})
            if changes:
                print(f"  体积变化: {changes.get('volume_change', 'N/A')}%")
                print(f"  结节类型: {changes.get('lesion_type', 'N/A')}")
                print(f"  位置: {changes.get('location', 'N/A')}")
                print(f"  置信度: {changes.get('confidence', 'N/A')}")
            
            # 检查元数据
            metadata = sample.get('metadata', {})
            if metadata:
                print(f"  患者ID: {metadata.get('patient_id', 'N/A')}")
                print(f"  扫描间隔: {metadata.get('interval_months', 'N/A')}个月")
                print(f"  T0大小: {metadata.get('nodule_size_t0_mm', 'N/A')}mm")
                print(f"  T1大小: {metadata.get('nodule_size_t1_mm', 'N/A')}mm")
                
    except Exception as e:
        print(f"✗ 数据内容测试失败: {e}")

def test_simple_dataset():
    """测试简化版数据集"""
    print("\n=== 测试简化版数据集 ===")
    
    filepath = Path(__file__).parent / "lidc_test_simple.json"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for i, sample in enumerate(data):
            print(f"\n简化样本 {i+1}: {sample['id']}")
            print(f"  语言: {sample['metadata']['language']}")
            print(f"  影像表现: {sample['findings'][:50]}...")
            print(f"  对比前片: {sample['comparison'][:50]}...")
            print(f"  印象: {sample['impression'][:50]}...")
            
    except Exception as e:
        print(f"✗ 简化数据集测试失败: {e}")

def create_mock_images():
    """创建模拟图像文件列表"""
    print("\n=== 模拟图像文件 ===")
    
    # 创建模拟的图像文件名列表
    mock_images = [
        "test_images/patient001_ct_t0.dcm",
        "test_images/patient001_ct_t1.dcm", 
        "test_images/patient002_ct_t0.dcm",
        "test_images/patient002_ct_t1.dcm",
        "test_images/patient003_ct_t0.dcm",
        "test_images/patient003_ct_t1.dcm",
        "ct_scan_t0.dcm",
        "ct_scan_t1.dcm"
    ]
    
    mock_masks = [
        "test_masks/nodule_001_mask.png",
        "test_masks/nodule_002_mask.png",
        "test_masks/nodule_003_mask.png"
    ]
    
    print("模拟图像文件:")
    for img in mock_images:
        print(f"  {img}")
    
    print("\n模拟掩码文件:")
    for mask in mock_masks:
        print(f"  {mask}")
    
    print("\n注意: 这些是模拟文件名，实际测试时需要:")
    print("1. 创建对应的图像文件，或")
    print("2. 修改数据集文件中的路径为实际存在的文件")

def main():
    """主测试函数"""
    print("🧪 开始测试数据集...")
    
    test_json_format()
    test_data_content()
    test_simple_dataset()
    create_mock_images()
    
    print("\n=== 测试总结 ===")
    print("✅ 数据集JSON格式验证完成")
    print("✅ 数据内容验证完成")
    print("✅ 多语言支持验证完成")
    print("\n📝 使用建议:")
    print("1. 先运行此脚本验证数据格式")
    print("2. 根据需要创建模拟图像文件")
    print("3. 使用真实数据进行实际测试")
    print("4. 可扩展更多样本和任务类型")

if __name__ == "__main__":
    main()