# LUNA16数据集使用示例
import json
import pandas as pd
from pathlib import Path

# 加载数据集
dataset_path = Path("datasets/luna16_hf")

# 读取样本数据
with open(dataset_path / "samples.json", 'r') as f:
    samples = json.load(f)

# 读取标注数据
annotations = pd.read_csv(dataset_path / "annotations.csv")

# 读取候选数据
candidates = pd.read_csv(dataset_path / "candidates.csv")

print(f"加载了 {len(samples)} 个样本")
print(f"包含 {len(annotations)} 个标注结节")
print(f"包含 {len(candidates)} 个候选结果")

# 显示第一个样本
if samples:
    first_sample = samples[0]
    print(f"\n第一个样本信息:")
    print(f"ID: {first_sample['id']}")
    print(f"结节数量: {len(first_sample['nodules'])}")
    if first_sample['nodules']:
        print(f"第一个结节直径: {first_sample['nodules'][0]['diameter_mm']:.2f} mm")
