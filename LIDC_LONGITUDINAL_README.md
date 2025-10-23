# LIDC-IDRI纵向推理分割任务实现

## 概述

本项目基于SegLLM框架实现了LIDC-IDRI纵向推理分割任务，支持对同一患者不同时间点的CT扫描进行变化分析和推理分割。

## 核心特性

### 1. 双时相图像处理
- 支持同一患者的基线(t0)和随访(t1)CT扫描输入
- 使用冻结的CLIP-ViT分别处理两张图像
- 通过特征拼接或差分实现图像比较
- 零参数增加，与原有LLM流程100%兼容

### 2. 四种推理任务类型

#### T1: 体积阈值推理
- **指令示例**: "分割所有较上次体积增加超过25%的结节"
- **评估指标**: 实例级准确率 + Dice
- **实现**: 基于体积变化率的阈值判断

#### T2: 新发/消退病灶
- **指令示例**: "标出新出现的磨玻璃结节"
- **评估指标**: 检测F1 + 分割Dice
- **实现**: 基于病灶存在性变化的检测

#### T3: 密度/形态变化
- **指令示例**: "圈出边界由清晰变模糊的病灶"
- **评估指标**: 条件一致率
- **实现**: 基于HU值变化和边界梯度分析

#### T4: 多属性组合
- **指令示例**: "分割体积增加≥20%且密度≥150HU的结节"
- **评估指标**: 多条件满足率
- **实现**: 多条件逻辑组合判断

### 3. 变化计算指标
- **体积变化**: 支持百分比和绝对值变化
- **密度变化**: HU值差异分析
- **直径变化**: 基于等效球直径的计算
- **边界模糊度**: 基于梯度分析的边界清晰度变化

### 4. SegLLM兼容性
- 使用标准[SEG] token格式，适配SegLLM框架的分割解码机制
- 兼容SegLLM的掩码生成与评估流程

### 4. 评估体系
- **几何一致性**: Dice系数、Hausdorff距离95%
- **演变识别**: 实例级Progression/Stable/Regression分类
- **条件一致性**: 满足指令条件的体素比例
- **阈值偏差**: 预测变化与真实变化的MAE

## 文件结构

```
llava/
├── model/
│   └── longitudinal_arch.py          # 纵向模型架构扩展
├── conversation_longitudinal.py      # 纵向推理对话模板
├── metrics_longitudinal.py           # 变化计算和评估指标
├── train/
│   └── train_longitudinal.py        # 训练脚本
├── inference_longitudinal_demo.py    # 推理演示脚本
└── constants.py                      # 纵向推理相关常量

configs/
└── longitudinal_lidc_config.json     # 配置文件

data/
└── lidc_longitudinal_dataset.py     # 数据加载器
```

## 快速开始

### 1. 数据准备

```python
from llava.data.lidc_longitudinal_dataset import LIDCLongitudinalDataset

# 创建数据集
dataset = LIDCLongitudinalDataset(
    data_root="path/to/lidc/data",
    split="train",
    task_weights={
        "volume_threshold": 0.3,
        "new_lesion": 0.25,
        "density_change": 0.25,
        "combined_attributes": 0.2
    }
)
```

### 2. 模型训练

```bash
python llava/train/train_longitudinal.py \
    --data_path data/lidc_longitudinal \
    --output_dir checkpoints/longitudinal_lidc \
    --num_train_epochs 15 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5
```

### 3. 推理演示

#### 交互式模式
```bash
python llava/inference_longitudinal_demo.py \
    --model_path checkpoints/longitudinal_lidc \
    --mode interactive
```

#### 批量推理模式
```bash
python llava/inference_longitudinal_demo.py \
    --model_path checkpoints/longitudinal_lidc \
    --mode batch \
    --test_cases test_cases.json \
    --output_dir inference_results
```

## 使用示例

### 生成纵向推理指令

```python
from llava.conversation_longitudinal import LongitudinalConversationTemplates

templates = LongitudinalConversationTemplates()

# 生成体积变化指令
instruction, response = templates.generate_instruction(
    task_type="volume_threshold",
    image_t0_path="baseline_ct.nii.gz",
    image_t1_path="followup_ct.nii.gz",
    target_mask_path="target_mask.nii.gz",
    change_info={"volume_change": 30, "density_change": 100}
)

# 快速指令示例
instruction = "[SEG] 请识别体积增加≥25%的肺结节"
instruction = "[SEG] 请识别新出现或消失的肺结节"
instruction = "[SEG] 请识别密度或形态发生显著变化的肺结节"
instruction = "[SEG] 请识别体积或密度发生显著变化的肺结节"
```

### 计算变化指标

```python
from llava.metrics_longitudinal import ChangeCalculator

calculator = ChangeCalculator(voxel_spacing=(1.0, 1.0, 1.0))

# 计算所有变化指标
metrics = calculator.calculate_all_changes(
    ct_t0=baseline_ct_array,
    ct_t1=followup_ct_array,
    mask_t0=baseline_mask,
    mask_t1=followup_mask
)

print(f"Volume change: {metrics.volume_change_percent:.1f}%")
print(f"Density change: {metrics.density_change_hu:.1f} HU")
```

### 评估模型性能

```python
from llava.metrics_longitudinal import LongitudinalMetrics

evaluator = LongitudinalMetrics()

# 计算评估指标
results = evaluator(
    pred_masks=predicted_masks,
    target_masks=ground_truth_masks,
    change_metrics_list=change_metrics,
    instructions=instructions
)

print(f"Dice: {results['dice_mean']:.3f} ± {results['dice_std']:.3f}")
print(f"Condition consistency: {results['condition_consistency_mean']:.3f}")
```

## 技术细节

### 模型架构修改

1. **LongitudinalMetaModel**: 扩展LlavaMetaModel，添加change_projector
2. **特征融合**: 支持拼接和差分两种方式
3. **零参数增加**: 复用现有CLIP-ViT，仅添加轻量级投影器

### 数据处理流程

1. **病例配对**: 基于PatientID和扫描日期
2. **病灶配准**: 刚性+非刚性配准
3. **变化计算**: 体积、密度、形态等多维度分析
4. **指令生成**: 基于变化指标的模板化生成

### 推理流程

1. **双图像输入**: [IMAGE256:t0] [IMAGE256:t1] 指令
2. **特征提取**: CLIP-ViT提取patch token
3. **变化建模**: 特征拼接/差分 + 投影
4. **条件推理**: 基于指令的变化筛选
5. **掩码生成**: 分割满足条件的病灶

## 性能优化

### 工程实践
- **GPU加速**: 所有计算支持CUDA
- **批处理**: 支持批量推理
- **内存优化**: 梯度检查点和混合精度训练
- **缓存机制**: 预处理结果缓存

### 算法优化
- **快速配准**: 基于ITK的高效配准算法
- **并行计算**: 多进程数据加载
- **向量化**: NumPy向量化操作
- **稀疏计算**: 掩码区域限定计算

## 扩展性

### 新任务类型
可以通过修改`LongitudinalConversationTemplates`添加新的任务类型：

```python
def _initialize_templates(self):
    templates = {
        "new_task": TaskTemplate(
            task_type="new_task",
            instruction_template="[IMAGE256:{image_t0}] [IMAGE256:{image_t1}] 描述新任务",
            response_template="[MASK-DECODE:{image_t1}|INFERENCE|{target_mask}]",
            thresholds={"new_threshold": 0.5},
            description="新任务描述"
        )
    }
    return templates
```

### 新评估指标
可以通过扩展`LongitudinalMetrics`添加新的评估指标：

```python
def new_metric(self, pred_masks, target_masks, **kwargs):
    # 实现新的评估逻辑
    return {"new_metric": score}
```

## 注意事项

1. **数据格式**: 确保CT图像为DICOM或NIfTI格式，掩码为PNG或NIfTI格式
2. **GPU内存**: 建议使用至少16GB显存的GPU进行训练
3. **数据预处理**: CT图像需要正确的HU值转换和窗口化，使用肺窗参数（window_center=-600, window_width=1500）
4. **模型兼容性**: 基于LLaVA-Med模型，确保版本兼容
5. **评估指标**: 部分指标需要3D掩码数据支持
6. **SegLLM兼容性**: 使用标准[SEG] token格式，确保与SegLLM框架的分割解码机制兼容
7. **训练稳定性**: __getitem__方法已修复死循环问题，设置最大重试次数为10次

## 故障排除

### 常见问题

1. **内存不足**: 减小batch size或使用梯度检查点
2. **配准失败**: 检查CT数据质量和配准参数
3. **推理速度慢**: 启用混合精度和模型编译
4. **分割不准确**: 调整模型阈值或增加训练数据

### 调试建议

1. 使用小批量数据进行快速验证
2. 可视化中间结果（配准、变化热图等）
3. 检查日志文件中的详细错误信息
4. 使用模拟数据进行单元测试

### 1. 训练中断
- **问题**：数据加载异常导致训练中断
- **解决**：检查数据路径和格式，确保所有文件可访问。__getitem__方法已修复死循环问题，设置最大重试次数为10次

### 2. 推理结果异常
- **问题**：分割结果不准确或出现错误
- **解决**：检查输入图像质量和预处理参数，确保使用正确的CT窗口化参数（肺窗：window_center=-600, window_width=1500）

### 3. 内存不足
- **问题**：GPU内存不足导致训练失败
- **解决**：减小批大小或使用梯度累积

### 4. 评估指标异常
- **问题**：评估指标计算结果不合理
- **解决**：检查掩码数据格式和质量，确保使用正确的切片选择（最大投影而非中间层）

### 5. SegLLM兼容性问题
- **问题**：模型无法识别自定义指令格式
- **解决**：确保使用标准[SEG] token格式，避免使用[MASK-DECODE:...]等自定义占位符

## 许可证

本项目基于SegLLM框架，遵循原项目的许可证条款。