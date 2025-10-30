# 医学影像数据集使用示例
import json
import pandas as pd
from pathlib import Path

# 加载数据集信息
with open('datasets/all_datasets_info.json', 'r') as f:
    datasets_info = json.load(f)

print("可用的数据集:")
for dataset in datasets_info['datasets']:
    print(f"- {dataset['name']}: {dataset['type']} ({dataset['modality']})")

# 示例：加载LUNA16风格数据集
luna16_path = Path('datasets/luna16_style')
with open(luna16_path / 'ct_scans.json', 'r') as f:
    ct_scans = json.load(f)

annotations = pd.read_csv(luna16_path / 'annotations.csv')
candidates = pd.read_csv(luna16_path / 'candidates.csv')

print(f"\nLUNA16风格数据集统计:")
print(f"- CT扫描数量: {len(ct_scans)}")
print(f"- 肺结节数量: {len(annotations)}")
print(f"- 候选结果数量: {len(candidates)}")
print(f"- 正样本候选: {len(candidates[candidates['class'] == 1])}")
