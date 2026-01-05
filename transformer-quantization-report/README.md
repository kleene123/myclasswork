# Transformer 模型量化课程设计报告项目

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

本项目是一个完整的 Transformer 模型量化技术研究与对比分析课程设计项目。项目实现了多种量化方法，并对量化前后的模型进行全面对比评估，生成详细的课程设计报告。

### 主要特性

- ✅ **多种量化方法**：动态量化、静态量化、量化感知训练(QAT)、混合精度
- ✅ **全面评估**：准确率、模型大小、推理速度、内存占用
- ✅ **丰富可视化**：对比图表、雷达图、综合仪表板
- ✅ **自动化实验**：一键运行所有实验
- ✅ **完整报告**：自动生成课程设计报告

## 项目结构

```
transformer-quantization-report/
├── README.md                          # 项目说明文档
├── requirements.txt                   # 依赖包列表
├── setup.py                          # 项目安装配置
├── report/                           # 课程设计报告目录
│   ├── 课程设计报告.md               # 完整的课程设计报告
│   ├── figures/                      # 报告图片目录
│   │   ├── architecture/             # 架构图
│   │   ├── results/                  # 实验结果图
│   │   └── comparisons/              # 对比图表
│   └── tables/                       # 实验数据表格
├── docs/                             # 技术文档
│   ├── 理论基础.md                   # 量化理论详解
│   ├── 实验设计.md                   # 实验方案设计
│   ├── 使用说明.md                   # 项目使用指南
│   └── 参考文献.md                   # 参考文献列表
├── src/                              # 源代码目录
│   ├── models/                       # 模型定义
│   ├── quantization/                 # 量化实现模块
│   ├── evaluation/                   # 评估模块
│   ├── utils/                        # 工具函数
│   └── training/                     # 训练模块
├── experiments/                      # 实验脚本
│   ├── 01_baseline_training.py      # 实验1: 训练基线模型
│   ├── 02_dynamic_quantization.py   # 实验2: 动态量化
│   ├── 03_static_quantization.py    # 实验3: 静态量化
│   ├── 04_qat_experiment.py         # 实验4: 量化感知训练
│   ├── 05_mixed_precision.py        # 实验5: 混合精度
│   ├── 06_comprehensive_comparison.py # 实验6: 综合对比
│   └── run_all_experiments.py       # 运行所有实验
├── configs/                          # 配置文件
│   ├── model_config.yaml            # 模型配置
│   ├── quantization_config.yaml     # 量化配置
│   ├── training_config.yaml         # 训练配置
│   └── experiment_config.yaml       # 实验配置
├── notebooks/                        # Jupyter notebooks
├── results/                          # 实验结果目录
├── scripts/                          # 辅助脚本
│   ├── generate_report.py           # 自动生成报告
│   ├── export_results.py            # 导出实验结果
│   └── create_visualizations.py     # 批量生成可视化图表
└── tests/                            # 单元测试
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

### 安装

1. 克隆仓库

```bash
git clone https://github.com/yourusername/transformer-quantization-report.git
cd transformer-quantization-report
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

或使用开发模式安装：

```bash
pip install -e .
```

### 运行实验

#### 方式1：运行所有实验（推荐）

```bash
cd experiments
python run_all_experiments.py
```

这将自动运行所有实验，并生成完整的报告。

#### 方式2：逐个运行实验

```bash
# 1. 训练基线模型
python experiments/01_baseline_training.py

# 2. 动态量化
python experiments/02_dynamic_quantization.py

# 3. 静态量化
python experiments/03_static_quantization.py

# 4. 量化感知训练
python experiments/04_qat_experiment.py

# 5. 混合精度
python experiments/05_mixed_precision.py

# 6. 综合对比
python experiments/06_comprehensive_comparison.py
```

#### 方式3：使用命令行工具

```bash
# 运行所有实验
run-experiments

# 生成报告
generate-report
```

### 查看结果

实验结果保存在 `results/` 目录下：

- `results/comparison/` - 对比图表和总结表格
- `results/baseline/` - 基线模型结果
- `results/dynamic_quant/` - 动态量化结果
- `results/static_quant/` - 静态量化结果
- `results/qat/` - QAT 结果
- `results/mixed_precision/` - 混合精度结果

课程设计报告位于 `report/课程设计报告.md`。

## 实验内容

### 1. 基线模型训练

训练未量化的 BERT 模型作为基线，用于后续对比。

### 2. 动态量化

对模型应用动态量化（Dynamic Quantization），主要量化 Linear 层的权重。

**特点**：
- 实现简单，无需额外训练
- 推理时动态计算激活值量化参数
- 适合 RNN、LSTM、Transformer 等模型

### 3. 静态量化

对模型应用静态量化（Static Quantization），需要校准数据集。

**特点**：
- 需要校准过程
- 量化权重和激活值
- 性能提升更明显

### 4. 量化感知训练（QAT）

在训练过程中模拟量化效果，获得更高精度的量化模型。

**特点**：
- 需要重新训练
- 准确率损失最小
- 训练时间较长

### 5. 混合精度

使用 FP16 或 INT8+FP16 混合精度加速推理。

**特点**：
- 灵活选择不同层的精度
- 平衡准确率和性能

### 6. 综合对比

对所有量化方法进行全面对比，生成对比图表和报告。

## 评估指标

- **准确率（Accuracy）**：模型分类准确率
- **F1 分数（F1 Score）**：精确率和召回率的调和平均
- **模型大小（Model Size）**：模型文件大小（MB）
- **压缩比（Compression Ratio）**：原始模型大小 / 量化模型大小
- **推理时间（Inference Time）**：单个批次的平均推理时间（ms）
- **加速比（Speedup）**：基线推理时间 / 量化模型推理时间
- **内存占用（Memory Usage）**：推理时的内存占用（MB）

## 配置说明

### 实验配置（experiment_config.yaml）

```yaml
experiments:
  baseline:
    enabled: true
    model: "bert-base-uncased"
    dataset: "imdb"
  
  dynamic_quantization:
    enabled: true
    dtype: "qint8"
    
evaluation:
  metrics:
    - accuracy
    - f1_score
    - model_size
    - inference_time
    - memory_usage
```

可以根据需要修改配置文件来自定义实验参数。

## 可视化示例

项目自动生成以下可视化图表：

1. **准确率对比图** - 展示各模型的准确率
2. **模型大小对比图** - 展示模型压缩效果
3. **推理速度对比图** - 展示推理性能提升
4. **综合性能雷达图** - 多维度性能对比
5. **准确率-大小权衡图** - 展示准确率与模型大小的权衡
6. **综合仪表板** - 一页展示所有关键指标

## 技术文档

详细的技术文档位于 `docs/` 目录：

- **理论基础.md** - 量化技术的数学原理和理论基础
- **实验设计.md** - 实验方案的详细设计
- **使用说明.md** - 详细的使用指南和 API 文档
- **参考文献.md** - 相关论文和资料

## 开发指南

### 添加新的量化方法

1. 在 `src/quantization/` 中创建新的量化模块
2. 在 `experiments/` 中创建对应的实验脚本
3. 在 `configs/quantization_config.yaml` 中添加配置
4. 在 `run_all_experiments.py` 中注册新实验

### 添加新的评估指标

1. 在 `src/evaluation/` 中实现评估函数
2. 在 `src/utils/metrics.py` 中添加指标计算
3. 在 `comparison.py` 中集成新指标

## 常见问题

### Q: 运行实验需要多长时间？

A: 完整运行所有实验大约需要 2-4 小时（取决于硬件配置）。可以通过修改 `experiment_config.yaml` 减少数据量以加快速度。

### Q: 是否需要 GPU？

A: 不是必需的，但强烈推荐使用 GPU 以加快训练和推理速度。

### Q: 如何使用自己的数据集？

A: 修改 `src/utils/data_loader.py` 中的数据加载函数，并在配置文件中指定新的数据集。

### Q: 生成的报告在哪里？

A: Markdown 报告位于 `report/课程设计报告.md`，可以使用 Pandoc 转换为 PDF。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 参考文献

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Jacob, B., et al. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." CVPR 2018.
3. Krishnamoorthi, R. "Quantizing deep convolutional networks for efficient inference: A whitepaper." arXiv 2018.
4. PyTorch Quantization Documentation: https://pytorch.org/docs/stable/quantization.html

## 联系方式

如有问题，请通过 Issue 或邮件联系。

---

**注意**：本项目仅用于教学和研究目的。
