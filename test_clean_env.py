#!/usr/bin/env python3
"""
完全清理环境的测试脚本
"""
import os
import sys

# 完全重置环境
original_path = sys.path.copy()
original_env = os.environ.copy()

print("=== 清理前环境 ===")
print(f"Python: {sys.executable}")
print(f"Python版本: {sys.version}")
print(f"路径数量: {len(sys.path)}")

# 清理sys.path - 只保留基本路径
sys.path = [
    path for path in sys.path 
    if not any(keyword in path.lower() for keyword in [
        'detectron2', 'uninext', 'segllm', 'llava', 'hipie'
    ])
]

# 清理环境变量
for key in list(os.environ.keys()):
    if any(keyword in key.lower() for keyword in ['pythonpath', 'detectron2']):
        del os.environ[key]

print("\n=== 清理后环境 ===")
print(f"路径数量: {len(sys.path)}")
print("主要路径:")
for p in sys.path[:5]:
    print(f"  {p}")

# 尝试导入
try:
    import torch
    print(f"\n✓ PyTorch {torch.__version__} 导入成功")
    print(f"✓ CUDA可用: {torch.cuda.is_available()}")
    
    # 创建简单模型测试
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    x = torch.randn(2, 10)
    y = model(x)
    print(f"✓ 模型前向传播成功，输出形状: {y.shape}")
    
except Exception as e:
    print(f"\n✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")