#!/usr/bin/env python3
"""
隔离测试脚本 - 完全避免detectron2冲突
"""
import os
import sys

# 清除所有detectron2相关路径
clean_paths = []
for path in sys.path:
    if 'detectron2' not in path.lower() and 'uninext' not in path.lower():
        clean_paths.append(path)

sys.path = clean_paths

# 设置隔离环境
os.environ['DETECTRON2_ENV'] = 'disabled'

# 尝试基础导入
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    # 尝试创建简单模型
    model = torch.nn.Linear(10, 5)
    print("✓ Basic PyTorch model created successfully")
    
    # 尝试张量操作
    x = torch.randn(1, 10)
    y = model(x)
    print(f"✓ Forward pass successful, output shape: {y.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n隔离测试完成")