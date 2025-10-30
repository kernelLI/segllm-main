#!/usr/bin/env python3
"""
极简测试脚本 - 避免任何可能的导入冲突
"""
import sys
import os

# 完全重置Python环境
print("=== 极简环境测试 ===")

# 尝试最基础的Python功能
print(f"Python版本: {sys.version}")
print(f"平台: {sys.platform}")

# 测试基本数学运算
result = 2 + 3
print(f"基本运算测试: 2 + 3 = {result}")

# 测试列表和字典
my_list = [1, 2, 3, 4, 5]
my_dict = {'a': 1, 'b': 2}
print(f"列表测试: {len(my_list)} 个元素")
print(f"字典测试: {len(my_dict)} 个键值对")

# 尝试延迟导入torch
print("\n=== 延迟导入测试 ===")
try:
    # 在最后一刻才导入torch
    import torch
    print(f"✓ PyTorch {torch.__version__} 导入成功")
    
    # 简单测试
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2
    print(f"✓ 张量运算成功: {x} -> {y}")
    
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")