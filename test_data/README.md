# 测试数据集说明

本目录包含了两个小型测试数据集，专门为本项目的纵向肺结节分析任务设计。

## 📁 数据集文件

### 1. `lidc_test_sample.json` - 完整版测试数据
**大小**: 3个样本 (约3KB)
**用途**: 测试完整的纵向分析功能
**格式**: 符合 `LIDCLongitudinalDataset` 类要求

**数据字段**:
- `id`: 样本唯一标识
- `image_t0/image_t1`: 前后两次CT扫描图像路径
- `target_mask`: 目标结节掩码路径
- `mask_paths`: 掩码路径列表
- `task_type`: 任务类型 (`volume_threshold` 或 `new_lesion`)
- `changes`: 变化信息 (体积变化、结节类型、位置等)
- `metadata`: 元数据 (患者ID、扫描日期、结节大小等)

### 2. `lidc_test_simple.json` - 简化版测试数据
**大小**: 3个样本 (约2KB)
**用途**: 测试简化版数据集功能
**格式**: 符合 `LIDCLongitudinalDatasetSimple` 类要求

**数据字段**:
- `id`: 样本唯一标识
- `image_paths`: CT图像路径列表
- `findings`: 影像表现描述
- `comparison`: 对比前片描述
- `impression`: 印象结论
- `metadata`: 元数据 (患者ID、扫描日期、语言等)

## 🧪 测试样本说明

### 样本001 - 稳定结节 (中文)
- 右肺上叶磨玻璃结节，8mm，无明显变化
- 任务类型: 体积阈值分析
- 预期: 良性病变，建议随访

### 样本002 - 新发结节 (英文)
- 左肺下叶新发实性结节，12mm
- 任务类型: 新发病灶检测
- 预期: 需要短期复查

### 样本003 - 增大结节 (中文)
- 左肺下叶部分实性结节，有增大趋势
- 任务类型: 体积阈值分析
- 预期: 需要进一步检查

## 🚀 使用示例

```python
# 测试完整版数据集
from llava.data.lidc_longitudinal_dataset import LIDCLongitudinalDataset

dataset = LIDCLongitudinalDataset(
    data_path="test_data/lidc_test_sample.json",
    image_root="test_data/images",
    tokenizer=tokenizer,
    data_args=data_args,
    image_processor=image_processor,
    language="chinese"
)

# 测试简化版数据集
from llava.data.lidc_longitudinal_dataset_simple import LIDCLongitudinalDatasetSimple

dataset = LIDCLongitudinalDatasetSimple(
    data_path="test_data/lidc_test_simple.json",
    language="chinese"
)
```

## 📊 数据集特点

✅ **小型化**: 仅3个样本，快速测试
✅ **多语言**: 支持中英文切换
✅ **完整性**: 覆盖主要任务类型
✅ **真实性**: 模拟真实临床场景
✅ **易扩展**: 可按需添加更多样本

## 🔧 扩展建议

如需更多测试数据，可以:
1. 复制现有样本并修改参数
2. 添加新的任务类型
3. 增加更多语言版本
4. 创建更复杂的病例场景