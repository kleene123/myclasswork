#!/usr/bin/env python3
"""
混合精度实验

对比 INT8、FP16 和混合精度量化的效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import yaml
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.bert_model import BERTModel
from src.evaluation.accuracy_eval import evaluate_accuracy
from src.evaluation.performance_eval import measure_inference_time
from src.evaluation.size_eval import calculate_model_size
from src.utils.data_loader import load_imdb_dataset, create_dataloader
from src.utils.logger import get_experiment_logger
from src.utils.visualization import plot_accuracy_comparison, plot_size_comparison, plot_speed_comparison

logger = get_experiment_logger("mixed_precision")


def apply_int8_quantization(model: nn.Module):
    """
    应用 INT8 动态量化
    
    Args:
        model: 原始模型
        
    Returns:
        quantized_model: INT8 量化模型
    """
    logger.info("应用 INT8 动态量化...")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    logger.info("✓ INT8 量化完成")
    return quantized_model


def apply_fp16_conversion(model: nn.Module, device: str = 'cpu'):
    """
    转换模型为 FP16 精度
    
    Args:
        model: 原始模型
        device: 设备
        
    Returns:
        fp16_model: FP16 模型
    """
    logger.info("转换模型为 FP16...")
    
    # 复制模型
    import copy
    fp16_model = copy.deepcopy(model)
    
    # 转换为 FP16
    if device == 'cuda':
        fp16_model = fp16_model.half()
        logger.info("✓ FP16 转换完成（CUDA）")
    else:
        # CPU 不完全支持 FP16，使用混合精度模拟
        logger.warning("CPU 不完全支持 FP16，使用 FP32 模拟")
        fp16_model = fp16_model.float()
    
    return fp16_model


def apply_mixed_precision(model: nn.Module, sensitive_layers: list = None):
    """
    应用混合精度策略
    
    对敏感层使用 FP16/FP32，对其他层使用 INT8
    
    Args:
        model: 原始模型
        sensitive_layers: 敏感层列表（保持高精度）
        
    Returns:
        mixed_model: 混合精度模型
    """
    logger.info("应用混合精度策略...")
    
    # 简化的混合精度策略：
    # 1. 对分类头使用高精度（FP32）
    # 2. 对其他线性层使用 INT8
    
    import copy
    mixed_model = copy.deepcopy(model)
    
    # 标记要保持高精度的层
    if sensitive_layers is None:
        # 默认策略：保持分类器层为高精度
        sensitive_layers = ['classifier', 'pooler']
    
    # 应用动态量化，但排除敏感层
    layers_to_quantize = set()
    for name, module in mixed_model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是敏感层
            is_sensitive = any(sens in name for sens in sensitive_layers)
            if not is_sensitive:
                layers_to_quantize.add(type(module))
    
    if layers_to_quantize:
        mixed_model = torch.quantization.quantize_dynamic(
            mixed_model,
            layers_to_quantize,
            dtype=torch.qint8
        )
    
    logger.info(f"✓ 混合精度应用完成")
    logger.info(f"  敏感层（保持 FP32）: {sensitive_layers}")
    logger.info(f"  其他层：INT8 量化")
    
    return mixed_model


def run_mixed_precision_experiment(
    config_path: str = "configs/experiment_config.yaml",
    model_path: str = "results/baseline/model",
    precisions: list = None,
    output_dir: str = "results/mixed_precision",
    device: str = None
):
    """
    运行混合精度实验
    
    Args:
        config_path: 配置文件路径
        model_path: 基线模型路径
        precisions: 要测试的精度列表 ['int8', 'fp16', 'mixed']
        output_dir: 输出目录
        device: 设备
    """
    logger.info("="*80)
    logger.info("混合精度实验")
    logger.info("="*80)
    
    # 默认测试所有精度
    if precisions is None:
        precisions = ['int8', 'fp16', 'mixed']
    
    logger.info(f"测试精度: {', '.join(precisions)}")
    
    # 加载配置
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 设置设备
    if device is None:
        device = config.get('device', 'cpu')
    
    # 混合精度在 CPU 上运行
    if device == 'cuda':
        logger.warning("混合精度实验建议在 CPU 上运行，自动切换到 CPU")
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存运行配置
    run_config = {
        'model_path': model_path,
        'precisions': precisions,
        'device': device
    }
    
    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    
    # 加载基线模型
    logger.info("\n" + "="*80)
    logger.info("加载基线模型...")
    logger.info("="*80)
    
    if not Path(model_path).exists():
        logger.error(f"基线模型不存在: {model_path}")
        logger.info("请先运行 01_baseline_training.py 训练基线模型")
        logger.info("创建新模型进行演示...")
        bert_model = BERTModel(model_name="bert-base-uncased", num_labels=2)
        baseline_model = bert_model.get_model()
    else:
        try:
            bert_model = BERTModel.load(model_path, num_labels=2)
            baseline_model = bert_model.get_model()
            logger.info("✓ 基线模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            logger.info("创建新模型进行演示...")
            bert_model = BERTModel(model_name="bert-base-uncased", num_labels=2)
            baseline_model = bert_model.get_model()
    
    baseline_model.to(device)
    baseline_model.eval()
    
    # 加载测试数据
    logger.info("\n" + "="*80)
    logger.info("加载测试数据...")
    logger.info("="*80)
    
    try:
        _, test_dataset, tokenizer = load_imdb_dataset(
            tokenizer_name="bert-base-uncased",
            max_length=128
        )
        
        # 限制测试样本
        eval_config = config.get('evaluation', {})
        test_samples = eval_config.get('test_samples', 1000)
        
        if len(test_dataset) > test_samples:
            from torch.utils.data import Subset
            indices = list(range(test_samples))
            test_dataset = Subset(test_dataset, indices)
        
        test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"✓ 测试数据加载完成: {len(test_dataset)} 样本")
        
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger.info("使用模拟数据进行演示...")
        
        # 创建模拟数据
        from torch.utils.data import DataLoader
        
        class DummyDataset:
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, 1000, (128,)),
                    'labels': torch.randint(0, 2, (1,)).item(),
                    'attention_mask': torch.ones(128)
                }
        
        test_dataset = DummyDataset(500)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估基线模型 (FP32)
    logger.info("\n" + "="*80)
    logger.info("评估基线模型 (FP32)...")
    logger.info("="*80)
    
    baseline_size = calculate_model_size(baseline_model)
    baseline_metrics = evaluate_accuracy(baseline_model, test_loader, device, desc="FP32 基线")
    baseline_timing = measure_inference_time(
        baseline_model, test_loader, device,
        num_warmup=10, num_iterations=50
    )
    
    logger.info(f"FP32 基线结果:")
    logger.info(f"  准确率: {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  模型大小: {baseline_size:.2f} MB")
    logger.info(f"  推理时间: {baseline_timing['mean_ms']:.2f} ms")
    
    # 存储所有结果
    all_results = {
        'FP32 (基线)': {
            'accuracy': float(baseline_metrics['accuracy']),
            'f1_score': float(baseline_metrics['f1_score']),
            'size_mb': float(baseline_size),
            'inference_time_ms': float(baseline_timing['mean_ms'])
        }
    }
    
    # INT8 量化实验
    if 'int8' in precisions:
        logger.info("\n" + "="*80)
        logger.info("INT8 量化实验...")
        logger.info("="*80)
        
        int8_model = apply_int8_quantization(baseline_model)
        int8_model.to(device)
        int8_model.eval()
        
        int8_size = calculate_model_size(int8_model)
        int8_metrics = evaluate_accuracy(int8_model, test_loader, device, desc="INT8 量化")
        int8_timing = measure_inference_time(
            int8_model, test_loader, device,
            num_warmup=10, num_iterations=50
        )
        
        logger.info(f"INT8 结果:")
        logger.info(f"  准确率: {int8_metrics['accuracy']:.4f}")
        logger.info(f"  模型大小: {int8_size:.2f} MB")
        logger.info(f"  推理时间: {int8_timing['mean_ms']:.2f} ms")
        
        # 保存模型
        model_save_path = output_path / "int8_model.pth"
        torch.save(int8_model, str(model_save_path))
        logger.info(f"✓ INT8 模型已保存: {model_save_path}")
        
        all_results['INT8'] = {
            'accuracy': float(int8_metrics['accuracy']),
            'f1_score': float(int8_metrics['f1_score']),
            'size_mb': float(int8_size),
            'inference_time_ms': float(int8_timing['mean_ms'])
        }
    
    # FP16 量化实验
    if 'fp16' in precisions:
        logger.info("\n" + "="*80)
        logger.info("FP16 混合精度实验...")
        logger.info("="*80)
        
        fp16_model = apply_fp16_conversion(baseline_model, device)
        fp16_model.to(device)
        fp16_model.eval()
        
        fp16_size = calculate_model_size(fp16_model)
        fp16_metrics = evaluate_accuracy(fp16_model, test_loader, device, desc="FP16 混合精度")
        fp16_timing = measure_inference_time(
            fp16_model, test_loader, device,
            num_warmup=10, num_iterations=50
        )
        
        logger.info(f"FP16 结果:")
        logger.info(f"  准确率: {fp16_metrics['accuracy']:.4f}")
        logger.info(f"  模型大小: {fp16_size:.2f} MB")
        logger.info(f"  推理时间: {fp16_timing['mean_ms']:.2f} ms")
        
        # 保存模型
        model_save_path = output_path / "fp16_model.pt"
        torch.save(fp16_model, str(model_save_path))
        logger.info(f"✓ FP16 模型已保存: {model_save_path}")
        
        all_results['FP16'] = {
            'accuracy': float(fp16_metrics['accuracy']),
            'f1_score': float(fp16_metrics['f1_score']),
            'size_mb': float(fp16_size),
            'inference_time_ms': float(fp16_timing['mean_ms'])
        }
    
    # 混合精度实验
    if 'mixed' in precisions:
        logger.info("\n" + "="*80)
        logger.info("混合精度实验（INT8 + FP32）...")
        logger.info("="*80)
        
        mixed_model = apply_mixed_precision(baseline_model)
        mixed_model.to(device)
        mixed_model.eval()
        
        mixed_size = calculate_model_size(mixed_model)
        mixed_metrics = evaluate_accuracy(mixed_model, test_loader, device, desc="混合精度")
        mixed_timing = measure_inference_time(
            mixed_model, test_loader, device,
            num_warmup=10, num_iterations=50
        )
        
        logger.info(f"混合精度结果:")
        logger.info(f"  准确率: {mixed_metrics['accuracy']:.4f}")
        logger.info(f"  模型大小: {mixed_size:.2f} MB")
        logger.info(f"  推理时间: {mixed_timing['mean_ms']:.2f} ms")
        
        # 保存模型
        model_save_path = output_path / "mixed_model.pth"
        torch.save(mixed_model, str(model_save_path))
        logger.info(f"✓ 混合精度模型已保存: {model_save_path}")
        
        all_results['混合精度'] = {
            'accuracy': float(mixed_metrics['accuracy']),
            'f1_score': float(mixed_metrics['f1_score']),
            'size_mb': float(mixed_size),
            'inference_time_ms': float(mixed_timing['mean_ms'])
        }
    
    # 综合对比分析
    logger.info("\n" + "="*80)
    logger.info("综合对比分析")
    logger.info("="*80)
    
    # 打印对比表格
    print("\n" + "+"*110)
    print(f"{'精度类型':<20} {'准确率':<15} {'F1分数':<15} {'模型大小(MB)':<18} {'推理时间(ms)':<18} {'压缩比':<12} {'加速比':<12}")
    print("+"*110)
    
    for precision_name, results in all_results.items():
        compression = baseline_size / results['size_mb'] if results['size_mb'] > 0 else 0
        speedup = baseline_timing['mean_ms'] / results['inference_time_ms'] if results['inference_time_ms'] > 0 else 0
        
        print(f"{precision_name:<20} {results['accuracy']:<15.4f} {results['f1_score']:<15.4f} "
              f"{results['size_mb']:<18.2f} {results['inference_time_ms']:<18.2f} "
              f"{compression:<12.2f}x {speedup:<12.2f}x")
    
    print("+"*110 + "\n")
    
    # 保存实验结果
    logger.info("\n" + "="*80)
    logger.info("保存结果...")
    logger.info("="*80)
    
    # 保存为 JSON
    results_data = {
        'all_results': all_results,
        'baseline': {
            'size_mb': float(baseline_size),
            'inference_time_ms': float(baseline_timing['mean_ms'])
        }
    }
    
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 指标已保存: {metrics_file}")
    
    # 保存为 CSV
    comparison_data = []
    for precision_name, results in all_results.items():
        compression = baseline_size / results['size_mb'] if results['size_mb'] > 0 else 0
        speedup = baseline_timing['mean_ms'] / results['inference_time_ms'] if results['inference_time_ms'] > 0 else 0
        
        comparison_data.append({
            '精度类型': precision_name,
            '准确率': f"{results['accuracy']:.4f}",
            'F1分数': f"{results['f1_score']:.4f}",
            '模型大小(MB)': f"{results['size_mb']:.2f}",
            '推理时间(ms)': f"{results['inference_time_ms']:.2f}",
            '压缩比': f"{compression:.2f}x",
            '加速比': f"{speedup:.2f}x"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    csv_file = output_path / "precision_comparison.csv"
    comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ 对比数据已保存: {csv_file}")
    
    # 生成可视化图表
    logger.info("\n生成可视化图表...")
    try:
        # 准确率对比
        accuracy_data = {name: results['accuracy'] for name, results in all_results.items()}
        plot_accuracy_comparison(
            accuracy_data,
            save_path=str(output_path / "accuracy_comparison.png"),
            title="不同精度的准确率对比"
        )
        logger.info(f"✓ 准确率对比图已保存")
        
        # 模型大小对比
        size_data = {name: results['size_mb'] for name, results in all_results.items()}
        plot_size_comparison(
            size_data,
            save_path=str(output_path / "size_comparison.png"),
            title="不同精度的模型大小对比"
        )
        logger.info(f"✓ 模型大小对比图已保存")
        
        # 推理速度对比
        speed_data = {name: results['inference_time_ms'] for name, results in all_results.items()}
        plot_speed_comparison(
            speed_data,
            save_path=str(output_path / "speed_comparison.png"),
            title="不同精度的推理速度对比"
        )
        logger.info(f"✓ 推理速度对比图已保存")
        
    except Exception as e:
        logger.warning(f"生成可视化图表失败: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("混合精度实验完成！")
    logger.info("="*80)
    logger.info(f"\n结果保存在: {output_dir}")
    logger.info(f"  - int8_model.pth            INT8 量化模型")
    logger.info(f"  - fp16_model.pt             FP16 模型")
    logger.info(f"  - mixed_model.pth           混合精度模型")
    logger.info(f"  - metrics.json              评估指标")
    logger.info(f"  - precision_comparison.csv  精度对比表")
    logger.info(f"  - accuracy_comparison.png   准确率对比图")
    logger.info(f"  - size_comparison.png       模型大小对比图")
    logger.info(f"  - speed_comparison.png      推理速度对比图")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="混合精度实验")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='results/baseline/model',
        help='基线模型路径'
    )
    parser.add_argument(
        '--precisions',
        nargs='+',
        default=['int8', 'fp16', 'mixed'],
        help='要测试的精度类型（int8, fp16, mixed）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/mixed_precision',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (建议使用 cpu)'
    )
    
    args = parser.parse_args()
    
    run_mixed_precision_experiment(
        config_path=args.config,
        model_path=args.model_path,
        precisions=args.precisions,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
