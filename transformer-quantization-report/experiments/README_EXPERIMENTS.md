# 实验脚本使用指南

本目录包含完整的 Transformer 模型量化实验脚本。

## 实验脚本列表

1. **01_baseline_training.py** - 基线模型训练
2. **02_dynamic_quantization.py** - 动态量化实验
3. **03_static_quantization.py** - 静态量化实验
4. **04_qat_experiment.py** - 量化感知训练实验
5. **05_mixed_precision.py** - 混合精度实验
6. **06_comprehensive_comparison.py** - 综合对比分析

## 快速开始

### 运行单个实验

#### 1. 训练基线模型
```bash
python experiments/01_baseline_training.py \
    --model bert-base-uncased \
    --dataset imdb \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --output_dir results/baseline
```

#### 2. 动态量化
```bash
python experiments/02_dynamic_quantization.py \
    --baseline-model results/baseline/model \
    --output_dir results/dynamic_quant
```

#### 3. 静态量化
```bash
python experiments/03_static_quantization.py \
    --model_path results/baseline/model \
    --calibration_samples 1000 \
    --output_dir results/static_quant
```

#### 4. 量化感知训练（QAT）
```bash
python experiments/04_qat_experiment.py \
    --model_path results/baseline/model \
    --epochs 2 \
    --learning_rate 1e-5 \
    --output_dir results/qat
```

#### 5. 混合精度实验
```bash
python experiments/05_mixed_precision.py \
    --model_path results/baseline/model \
    --precisions int8 fp16 mixed \
    --output_dir results/mixed_precision
```

#### 6. 综合对比
```bash
python experiments/06_comprehensive_comparison.py \
    --results-dir results
```

### 运行所有实验

```bash
# 运行完整实验流程
python experiments/run_all_experiments.py

# 跳过基线训练（如果已有模型）
python experiments/run_all_experiments.py --skip-training

# 只运行综合对比
python experiments/run_all_experiments.py --only-comparison
```

## 输出文件结构

```
results/
├── baseline/
│   ├── model/                      # 训练好的模型
│   ├── config.json                 # 运行配置
│   ├── metrics.json                # 评估指标
│   ├── evaluation_results.csv      # 评估结果表
│   └── training_curve.png          # 训练曲线图
│
├── dynamic_quant/
│   ├── quantized_model.pt          # 动态量化模型
│   └── results.json                # 结果数据
│
├── static_quant/
│   ├── quantized_model.pt          # 静态量化模型
│   ├── config.json                 # 运行配置
│   ├── metrics.json                # 评估指标
│   └── comparison.csv              # 对比数据
│
├── qat/
│   ├── qat_model.pth               # QAT 量化模型
│   ├── config.json                 # 运行配置
│   ├── metrics.json                # 评估指标
│   ├── comparison_table.csv        # 三方对比表
│   └── training_curve.png          # 训练曲线图
│
├── mixed_precision/
│   ├── int8_model.pth              # INT8 模型
│   ├── fp16_model.pt               # FP16 模型
│   ├── mixed_model.pth             # 混合精度模型
│   ├── config.json                 # 运行配置
│   ├── metrics.json                # 评估指标
│   ├── precision_comparison.csv    # 精度对比表
│   ├── accuracy_comparison.png     # 准确率对比图
│   ├── size_comparison.png         # 模型大小对比图
│   └── speed_comparison.png        # 推理速度对比图
│
└── comparison/
    ├── summary_table.csv           # 综合对比表
    ├── summary_report.md           # 总结报告
    ├── accuracy_comparison.png     # 准确率对比图
    ├── size_comparison.png         # 模型大小对比图
    ├── speed_comparison.png        # 推理速度对比图
    ├── radar_chart.png             # 综合性能雷达图
    ├── accuracy_vs_size.png        # 准确率-大小权衡图
    └── summary_dashboard.png       # 综合仪表板
```

## 脚本详细说明

### 01_baseline_training.py

训练基线 BERT 模型用于文本分类任务。

**主要功能**：
- 加载 IMDB 数据集
- 训练 BERT 模型（3 epochs）
- 保存最佳模型
- 在测试集上评估性能
- 生成训练曲线图

**配置参数**：
- `--model`: 模型名称（默认: bert-base-uncased）
- `--dataset`: 数据集名称（默认: imdb）
- `--epochs`: 训练轮数（默认: 3）
- `--batch_size`: 批次大小（默认: 32）
- `--learning_rate`: 学习率（默认: 2e-5）

### 03_static_quantization.py

应用静态量化并评估效果。

**主要功能**：
- 加载基线模型
- 准备校准数据集（1000 样本）
- 应用静态量化（融合模块、插入观察器、校准、转换）
- 评估量化模型性能
- 计算压缩比和加速比

**配置参数**：
- `--model_path`: 基线模型路径（默认: results/baseline/model）
- `--calibration_samples`: 校准样本数量（默认: 1000）

### 04_qat_experiment.py

实现量化感知训练（QAT）。

**主要功能**：
- 加载预训练基线模型
- 插入伪量化节点准备 QAT 模型
- 进行 QAT 训练（2 epochs）
- 转换为真正的量化模型
- 三方对比（基线 vs 静态量化 vs QAT）

**配置参数**：
- `--model_path`: 基线模型路径（默认: results/baseline/model）
- `--epochs`: QAT 训练轮数（默认: 2）
- `--learning_rate`: 学习率（默认: 1e-5）

### 05_mixed_precision.py

测试不同精度配置的效果。

**主要功能**：
- INT8 动态量化实验
- FP16 混合精度实验
- 混合精度策略（INT8 + FP32）
- 多方对比分析
- 生成对比图表

**配置参数**：
- `--model_path`: 基线模型路径（默认: results/baseline/model）
- `--precisions`: 要测试的精度类型（默认: int8 fp16 mixed）

## 依赖项

确保已安装以下依赖：

```bash
pip install -r requirements.txt
```

主要依赖：
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0
- pyyaml >= 6.0
- scikit-learn >= 1.3.0

## 注意事项

1. **设备选择**：
   - 基线训练：支持 CPU/CUDA
   - 动态量化：支持 CPU/CUDA
   - 静态量化：仅支持 CPU
   - QAT：建议使用 CPU
   - 混合精度：建议使用 CPU

2. **内存要求**：
   - 基线训练需要约 8GB 内存
   - 量化实验需要约 4GB 内存

3. **训练时间**：
   - 基线训练：约 1-2 小时（3 epochs）
   - QAT 训练：约 30-60 分钟（2 epochs）
   - 其他实验：约 10-30 分钟

4. **数据集**：
   - 默认使用 IMDB 数据集（自动下载）
   - 首次运行会下载数据集到 `./data/cache/`

## 常见问题

### Q: 如何使用自定义配置？

A: 修改 `configs/training_config.yaml` 或通过命令行参数覆盖。

### Q: 如何跳过某个实验？

A: 单独运行需要的实验脚本，或修改 `run_all_experiments.py`。

### Q: 模型保存失败怎么办？

A: 检查输出目录权限，确保有足够的磁盘空间。

### Q: 如何查看实验日志？

A: 日志会输出到控制台，可以使用重定向保存：
```bash
python experiments/01_baseline_training.py > log.txt 2>&1
```

## 扩展实验

可以通过修改以下参数进行扩展实验：

1. **不同模型**：
   - DistilBERT: `--model distilbert-base-uncased`
   - RoBERTa: `--model roberta-base`

2. **不同数据集**：
   - SST-2: `--dataset sst2`
   - AG News: `--dataset ag_news`

3. **不同量化配置**：
   - 修改 `configs/quantization_config.yaml`

## 技术支持

如有问题，请查看：
- 项目文档：`docs/`
- 课程设计报告：`report/课程设计报告.md`
- 项目总结：`PROJECT_SUMMARY.md`
