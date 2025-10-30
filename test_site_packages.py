#!/usr/bin/env python3
"""
检查site-packages中的潜在冲突
"""
import os
import sys

site_packages = 'C:\\anaconda\\envs\\my_pytorch\\lib\\site-packages'

print("=== 检查site-packages中的冲突 ===")
print(f"路径: {site_packages}")

# 查找可能的detectron2相关文件
keywords = ['detectron2', 'group', 'spatial', 'softmax']

conflicts = []
if os.path.exists(site_packages):
    for root, dirs, files in os.walk(site_packages):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        if all(keyword in content for keyword in keywords):
                            conflicts.append(filepath)
                            print(f"发现潜在冲突: {filepath}")
                except:
                    continue

print(f"\n发现 {len(conflicts)} 个潜在冲突文件")

# 检查.pth文件
print("\n=== 检查.pth文件 ===")
pth_files = []
for file in os.listdir(site_packages):
    if file.endswith('.pth'):
        pth_files.append(file)
        print(f"发现.pth文件: {file}")
        
        # 读取.pth内容
        pth_path = os.path.join(site_packages, file)
        try:
            with open(pth_path, 'r') as f:
                content = f.read()
                print(f"  内容: {content.strip()}")
        except:
            print("  无法读取内容")

print(f"\n发现 {len(pth_files)} 个.pth文件")