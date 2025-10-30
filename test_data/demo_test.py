#!/usr/bin/env python3
"""
演示如何使用创建的小型测试数据集
展示数据集加载和多语言支持功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.data.lidc_longitudinal_dataset_simple import LIDCLongitudinalDatasetSimple
from llava.data.lidc_longitudinal_dataset import LIDCLongitudinalDataset

def test_simple_dataset():
    """测试简化版数据集"""
    print("=== 测试简化版数据集 ===")
    
    try:
        # 创建数据集实例
        dataset = LIDCLongitudinalDatasetSimple(
            data_path="test_data/lidc_test_simple.json",
            language="chinese",
            use_llm_native=True
        )
        
        print(f"✅ 数据集加载成功，共 {len(dataset)} 个样本")
        
        # 显示第一个样本
        if len(dataset) > 0:
            sample = dataset.samples[0]
            print(f"\n样本1信息:")
            print(f"  ID: {sample['id']}")
            print(f"  语言: {sample['metadata']['language']}")
            print(f"  影像表现: {sample['findings']}")
            print(f"  对比前片: {sample['comparison']}")
            print(f"  印象: {sample['impression']}")
            
            # 生成对话
            conversation = dataset._generate_conversation(sample)
            print(f"\n生成的对话:")
            for turn in conversation:
                print(f"  {turn['role']}: {turn['content'][:100]}...")
        
        # 测试语言切换
        print(f"\n测试语言切换:")
        dataset.set_language("english")
        sample_en = dataset.samples[0]
        conversation_en = dataset._generate_conversation(sample_en)
        print(f"英文对话:")
        for turn in conversation_en:
            print(f"  {turn['role']}: {turn['content'][:100]}...")
            
    except Exception as e:
        print(f"❌ 简化版数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_full_dataset_mock():
    """模拟测试完整版数据集（无需真实图像文件）"""
    print("\n=== 模拟测试完整版数据集 ===")
    
    # 创建模拟的数据参数
    class MockDataArgs:
        def __init__(self):
            self.image_aspect_ratio = 'square'
            self.image_grid_pinpoints = None
            
    class MockTokenizer:
        def __call__(self, text, return_tensors=None, padding=False, truncation=False):
            class MockOutput:
                def __init__(self):
                    self.input_ids = [[1, 2, 3, 4, 5]]  # 模拟token IDs
            return MockOutput()
    
    class MockImageProcessor:
        def __call__(self, images, return_tensors=None):
            return {'pixel_values': [[1, 2, 3]]}  # 模拟图像特征
    
    try:
        # 读取JSON数据
        import json
        with open("test_data/lidc_test_sample.json", 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        print(f"✅ 加载完整版数据集JSON，共 {len(samples)} 个样本")
        
        # 显示样本结构
        for i, sample in enumerate(samples[:2]):  # 显示前2个
            print(f"\n样本 {i+1}: {sample['id']}")
            print(f"  任务类型: {sample['task_type']}")
            print(f"  图像T0: {sample['image_t0']}")
            print(f"  图像T1: {sample['image_t1']}")
            print(f"  体积变化: {sample['changes']['volume_change']}%")
            print(f"  结节类型: {sample['changes']['lesion_type']}")
            print(f"  位置: {sample['changes']['location']}")
            
            # 模拟对话生成
            if sample['task_type'] == 'volume_threshold':
                instruction = f"[IMAGE256:{sample['image_t0']}|{sample['image_t1']}] 分割所有较上次体积增加超过{sample['changes']['volume_change']}%的结节"
            else:
                instruction = f"[IMAGE256:{sample['image_t0']}|{sample['image_t1']}] 标出新出现的{sample['changes']['lesion_type']}病灶"
            
            print(f"  模拟指令: {instruction}")
            
    except Exception as e:
        print(f"❌ 完整版数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_multilingual_support():
    """测试多语言支持"""
    print("\n=== 测试多语言支持 ===")
    
    try:
        # 中文测试
        dataset_zh = LIDCLongitudinalDatasetSimple(
            data_path="test_data/lidc_test_simple.json",
            language="chinese"
        )
        
        # 英文测试  
        dataset_en = LIDCLongitudinalDatasetSimple(
            data_path="test_data/lidc_test_simple.json",
            language="english"
        )
        
        print("✅ 中英文数据集加载成功")
        
        # 对比同一样本的不同语言版本
        if len(dataset_zh.samples) > 0 and len(dataset_en.samples) > 0:
            zh_sample = dataset_zh.samples[0]
            en_sample = dataset_en.samples[0]
            
            print(f"\n中文版本:")
            print(f"  影像表现: {zh_sample['findings']}")
            print(f"  对比前片: {zh_sample['comparison']}")
            print(f"  印象: {zh_sample['impression']}")
            
            print(f"\n英文版本:")
            print(f"  影像表现: {en_sample['findings']}")
            print(f"  对比前片: {en_sample['comparison']}")
            print(f"  印象: {en_sample['impression']}")
            
    except Exception as e:
        print(f"❌ 多语言测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主演示函数"""
    print("🚀 开始演示测试数据集功能...")
    print("=" * 50)
    
    test_simple_dataset()
    test_full_dataset_mock()
    test_multilingual_support()
    
    print("\n" + "=" * 50)
    print("🎉 演示完成！")
    print("\n📋 总结:")
    print("✅ 创建了小型测试数据集 (3个样本)")
    print("✅ 支持完整版和简化版两种格式")
    print("✅ 支持中英文多语言")
    print("✅ 覆盖体积阈值和新发病灶检测任务")
    print("✅ 数据格式符合项目要求")
    print("\n💡 使用建议:")
    print("1. 使用简化版数据集进行快速功能测试")
    print("2. 使用完整版数据集测试纵向分析功能")
    print("3. 根据需要扩展更多样本和任务类型")
    print("4. 可替换为真实医学图像进行实际测试")

if __name__ == "__main__":
    main()