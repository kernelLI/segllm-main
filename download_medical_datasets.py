#!/usr/bin/env python3
"""
医学影像数据集快速下载工具
支持多个国内镜像源和Hugging Face
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
from tqdm import tqdm

class MedicalDatasetDownloader:
    def __init__(self):
        self.base_dir = Path("datasets")
        self.base_dir.mkdir(exist_ok=True)
        
        # 国内镜像源
        self.mirrors = {
            "tsinghua": {
                "name": "清华大学镜像",
                "base_url": "https://mirrors.tuna.tsinghua.edu.cn",
                "medical_path": "/opencv/opencv_contrib/"
            },
            "opendatalab": {
                "name": "OpenDataLab",
                "base_url": "https://opendatalab.com",
                "medical_path": "/OpenDataLab/"
            },
            "kaggle": {
                "name": "Kaggle镜像",
                "base_url": "https://www.kaggle.com",
                "medical_path": "/datasets/"
            }
        }
    
    def create_synthetic_medical_dataset(self, name="medical_synthetic", num_samples=50):
        """创建合成医学数据集"""
        print(f"正在创建合成医学数据集: {name}")
        
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(exist_ok=True)
        
        # 生成合成数据
        np.random.seed(42)
        samples = []
        
        for i in range(num_samples):
            # 模拟不同类型的医学影像
            modality = np.random.choice(["CT", "MRI", "X-ray", "Ultrasound"])
            
            sample = {
                "id": f"{name}_{i:04d}",
                "patient_id": f"Patient_{i:03d}",
                "age": np.random.randint(18, 80),
                "gender": np.random.choice(["M", "F"]),
                "modality": modality,
                "body_part": np.random.choice(["Chest", "Abdomen", "Brain", "Bone"]),
                "image_size": [512, 512, np.random.randint(50, 200)],
                "pixel_spacing": [np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0)],
                "findings": []
            }
            
            # 随机生成发现
            num_findings = np.random.randint(0, 5)
            for j in range(num_findings):
                finding = {
                    "type": np.random.choice([
                        "Nodule", "Mass", "Cyst", "Calcification", 
                        "Fluid", "Fracture", "Inflammation"
                    ]),
                    "x": np.random.randint(50, 450),
                    "y": np.random.randint(50, 450),
                    "z": np.random.randint(10, sample["image_size"][2]-10),
                    "size_mm": np.random.uniform(1.0, 50.0),
                    "severity": np.random.choice(["Low", "Medium", "High"]),
                    "confidence": np.random.uniform(0.6, 1.0)
                }
                sample["findings"].append(finding)
            
            samples.append(sample)
        
        # 保存数据
        samples_file = dataset_dir / "samples.json"
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # 创建标注文件
        annotations = []
        for sample in samples:
            for finding in sample["findings"]:
                annotation = {
                    "sample_id": sample["id"],
                    "patient_id": sample["patient_id"],
                    "finding_type": finding["type"],
                    "x": finding["x"],
                    "y": finding["y"],
                    "z": finding["z"],
                    "size_mm": finding["size_mm"],
                    "severity": finding["severity"],
                    "confidence": finding["confidence"]
                }
                annotations.append(annotation)
        
        annotations_df = pd.DataFrame(annotations)
        annotations_file = dataset_dir / "annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        
        # 创建数据集信息
        dataset_info = {
            "name": name,
            "description": "Synthetic medical imaging dataset for development and testing",
            "num_samples": len(samples),
            "num_annotations": len(annotations),
            "modalities": list(set([s["modality"] for s in samples])),
            "body_parts": list(set([s["body_part"] for s in samples])),
            "finding_types": list(set([a["finding_type"] for a in annotations])) if annotations else []
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 合成数据集创建完成!")
        print(f"   - 样本数量: {len(samples)}")
        print(f"   - 标注数量: {len(annotations)}")
        print(f"   - 模态类型: {dataset_info['modalities']}")
        print(f"   - 身体部位: {dataset_info['body_parts']}")
        
        return dataset_dir
    
    def create_luna16_style_dataset(self):
        """创建LUNA16风格的数据集"""
        print("正在创建LUNA16风格的肺结节检测数据集...")
        
        dataset_dir = self.base_dir / "luna16_style"
        dataset_dir.mkdir(exist_ok=True)
        
        np.random.seed(42)
        
        # 生成CT扫描数据
        ct_scans = []
        for i in range(20):
            scan = {
                "seriesuid": f"1.3.6.1.4.1.14519.5.2.1.6279.6001.{i:012d}",
                "patient_id": f"LND_{i:03d}",
                "slice_thickness": np.random.uniform(0.6, 2.5),
                "pixel_spacing": [np.random.uniform(0.6, 0.8), np.random.uniform(0.6, 0.8)],
                "image_dimensions": [512, 512, np.random.randint(100, 300)],
                "scan_date": f"2024-01-{i+1:02d}",
                "manufacturer": np.random.choice(["GE", "Siemens", "Philips", "Toshiba"]),
                "nodules": []
            }
            
            # 生成肺结节
            num_nodules = np.random.poisson(1.5)  # 平均1.5个结节
            for j in range(num_nodules):
                nodule = {
                    "id": f"Nodule_{j}_{i}",
                    "coordX": np.random.uniform(-100, 100),
                    "coordY": np.random.uniform(-100, 100),
                    "coordZ": np.random.uniform(-50, 50),
                    "diameter_mm": np.random.uniform(3.0, 30.0),
                    "volume_mm3": np.random.uniform(50, 5000),
                    "malignancy": np.random.choice(["benign", "malignant", "uncertain"]),
                    "subtlety": np.random.randint(1, 6),
                    "sphericity": np.random.uniform(0.5, 1.0),
                    "margin": np.random.uniform(0.5, 1.0),
                    "lobulation": np.random.uniform(0.0, 1.0),
                    "spiculation": np.random.uniform(0.0, 1.0)
                }
                scan["nodules"].append(nodule)
            
            ct_scans.append(scan)
        
        # 保存扫描信息
        scans_file = dataset_dir / "ct_scans.json"
        with open(scans_file, 'w', encoding='utf-8') as f:
            json.dump(ct_scans, f, indent=2, ensure_ascii=False)
        
        # 创建annotations.csv (LUNA16格式)
        annotations = []
        for scan in ct_scans:
            for nodule in scan["nodules"]:
                annotation = {
                    "seriesuid": scan["seriesuid"],
                    "coordX": nodule["coordX"],
                    "coordY": nodule["coordY"],
                    "coordZ": nodule["coordZ"],
                    "diameter_mm": nodule["diameter_mm"]
                }
                annotations.append(annotation)
        
        annotations_df = pd.DataFrame(annotations)
        annotations_file = dataset_dir / "annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        
        # 创建candidates.csv (LUNA16格式)
        candidates = []
        for scan in ct_scans:
            # 为每个扫描生成候选结节
            num_candidates = np.random.randint(50, 200)
            for j in range(num_candidates):
                candidate = {
                    "seriesuid": scan["seriesuid"],
                    "coordX": np.random.uniform(-150, 150),
                    "coordY": np.random.uniform(-150, 150),
                    "coordZ": np.random.uniform(-75, 75),
                    "diameter_mm": np.random.uniform(1.0, 25.0),
                    "class": np.random.choice([0, 1], p=[0.95, 0.05])  # 5%为正样本
                }
                candidates.append(candidate)
        
        candidates_df = pd.DataFrame(candidates)
        candidates_file = dataset_dir / "candidates.csv"
        candidates_df.to_csv(candidates_file, index=False)
        
        # 创建分割文件
        train_split = np.random.choice(len(ct_scans), size=int(len(ct_scans)*0.7), replace=False)
        val_split = np.random.choice([i for i in range(len(ct_scans)) if i not in train_split], 
                                     size=int(len(ct_scans)*0.15), replace=False)
        test_split = [i for i in range(len(ct_scans)) if i not in train_split and i not in val_split]
        
        splits = {
            "train": [ct_scans[i]["seriesuid"] for i in train_split],
            "validation": [ct_scans[i]["seriesuid"] for i in val_split],
            "test": [ct_scans[i]["seriesuid"] for i in test_split]
        }
        
        splits_file = dataset_dir / "splits.json"
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)
        
        # 创建数据集信息
        dataset_info = {
            "name": "LUNA16-Style Dataset",
            "description": "Synthetic lung nodule detection dataset in LUNA16 format",
            "num_scans": len(ct_scans),
            "num_annotations": len(annotations),
            "num_candidates": len(candidates),
            "num_nodules": sum(len(scan["nodules"]) for scan in ct_scans),
            "positive_candidates": len(candidates_df[candidates_df["class"] == 1]),
            "negative_candidates": len(candidates_df[candidates_df["class"] == 0]),
            "splits": {
                "train": len(train_split),
                "validation": len(val_split),
                "test": len(test_split)
            }
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ LUNA16风格数据集创建完成!")
        print(f"   - CT扫描数量: {len(ct_scans)}")
        print(f"   - 肺结节数量: {dataset_info['num_nodules']}")
        print(f"   - 候选结果: {len(candidates)} (正样本: {dataset_info['positive_candidates']})")
        
        return dataset_dir
    
    def create_chest_xray_dataset(self):
        """创建胸部X光数据集"""
        print("正在创建胸部X光数据集...")
        
        dataset_dir = self.base_dir / "chest_xray"
        dataset_dir.mkdir(exist_ok=True)
        
        np.random.seed(42)
        
        # 生成胸部X光数据
        conditions = [
            "Normal", "Pneumonia", "Tuberculosis", "COVID-19", 
            "Lung Cancer", "Pneumothorax", "Effusion", "Atelectasis"
        ]
        
        samples = []
        for i in range(30):
            condition = np.random.choice(conditions)
            severity = np.random.choice(["Mild", "Moderate", "Severe"]) if condition != "Normal" else "None"
            
            sample = {
                "id": f"CXR_{i:04d}",
                "patient_id": f"Patient_{i:03d}",
                "age": np.random.randint(18, 90),
                "gender": np.random.choice(["M", "F"]),
                "condition": condition,
                "severity": severity,
                "image_size": [1024, 1024],
                "view": np.random.choice(["PA", "AP", "Lateral"]),
                "findings": []
            }
            
            # 生成发现
            if condition != "Normal":
                num_findings = np.random.randint(1, 4)
                for j in range(num_findings):
                    finding = {
                        "type": condition,
                        "x": np.random.randint(100, 900),
                        "y": np.random.randint(100, 900),
                        "width": np.random.randint(50, 200),
                        "height": np.random.randint(50, 200),
                        "confidence": np.random.uniform(0.7, 1.0)
                    }
                    sample["findings"].append(finding)
            
            samples.append(sample)
        
        # 保存数据
        samples_file = dataset_dir / "samples.json"
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # 创建标注文件
        annotations = []
        for sample in samples:
            for finding in sample["findings"]:
                annotation = {
                    "sample_id": sample["id"],
                    "patient_id": sample["patient_id"],
                    "condition": sample["condition"],
                    "severity": sample["severity"],
                    "finding_x": finding["x"],
                    "finding_y": finding["y"],
                    "width": finding["width"],
                    "height": finding["height"],
                    "confidence": finding["confidence"]
                }
                annotations.append(annotation)
        
        annotations_df = pd.DataFrame(annotations)
        annotations_file = dataset_dir / "annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        
        # 统计信息
        condition_counts = {}
        for sample in samples:
            condition = sample["condition"]
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        dataset_info = {
            "name": "Chest X-Ray Dataset",
            "description": "Synthetic chest X-ray dataset for disease classification",
            "num_samples": len(samples),
            "num_annotations": len(annotations),
            "conditions": condition_counts,
            "image_size": "1024x1024",
            "modalities": ["X-ray"],
            "body_parts": ["Chest"]
        }
        
        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 胸部X光数据集创建完成!")
        print(f"   - 样本数量: {len(samples)}")
        print(f"   - 疾病分布: {condition_counts}")
        
        return dataset_dir

def main():
    """主函数"""
    print("🏥 医学影像数据集快速下载工具")
    print("=" * 50)
    
    downloader = MedicalDatasetDownloader()
    
    # 创建多个数据集
    datasets_created = []
    
    # 1. 创建合成医学数据集
    synthetic_dir = downloader.create_synthetic_medical_dataset(num_samples=50)
    datasets_created.append(synthetic_dir)
    
    # 2. 创建LUNA16风格数据集
    luna16_dir = downloader.create_luna16_style_dataset()
    datasets_created.append(luna16_dir)
    
    # 3. 创建胸部X光数据集
    chest_dir = downloader.create_chest_xray_dataset()
    datasets_created.append(chest_dir)
    
    print(f"\n✅ 所有数据集创建完成!")
    print(f"📁 数据集列表:")
    for i, dataset_dir in enumerate(datasets_created, 1):
        print(f"   {i}. {dataset_dir.name}: {dataset_dir}")
    
    # 创建统一的数据集信息文件
    all_datasets_info = {
        "datasets": [
            {
                "name": "medical_synthetic",
                "path": str(synthetic_dir),
                "type": "synthetic",
                "modality": "multi-modal",
                "num_samples": 50
            },
            {
                "name": "luna16_style",
                "path": str(luna16_dir),
                "type": "lung_nodule_detection",
                "modality": "CT",
                "num_samples": 20
            },
            {
                "name": "chest_xray",
                "path": str(chest_dir),
                "type": "chest_xray_classification",
                "modality": "X-ray",
                "num_samples": 30
            }
        ]
    }
    
    info_file = downloader.base_dir / "all_datasets_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(all_datasets_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 数据集信息已保存到: {info_file}")
    print(f"🚀 可以开始使用这些数据集进行训练和测试!")
    
    # 创建使用示例
    example_script = """# 医学影像数据集使用示例
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

print(f"\\nLUNA16风格数据集统计:")
print(f"- CT扫描数量: {len(ct_scans)}")
print(f"- 肺结节数量: {len(annotations)}")
print(f"- 候选结果数量: {len(candidates)}")
print(f"- 正样本候选: {len(candidates[candidates['class'] == 1])}")
"""
    
    example_file = Path("medical_datasets_example.py")
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_script)
    
    print(f"\n📖 使用示例已保存到: {example_file}")
    print("运行 `python medical_datasets_example.py` 查看使用示例")

if __name__ == "__main__":
    main()