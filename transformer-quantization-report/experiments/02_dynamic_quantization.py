#!/usr/bin/env python3
"""
动态量化实验

对 BERT 模型应用动态量化并评估效果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import argparse

from src.models.bert_model import BERTModel
from src.quantization.dynamic_quantization import (
    apply_dynamic_quantization,
    save_quantized_model,
    compare_model_sizes
)
from src.evaluation.accuracy_eval import evaluate_accuracy
from src.evaluation.performance_eval import measure_inference_time
from src.evaluation.size_eval import calculate_model_size
from src.utils.data_loader import load_imdb_dataset, create_dataloader
from src.utils.logger import get_experiment_logger

logger = get_experiment_logger("dynamic_quantization")


def run_dynamic_quantization_experiment(
    config_path: str = "configs/experiment_config.yaml",
    results_dir: str = "results/dynamic_quant",
    baseline_model_path: str = "results/baseline/model",
    device: str = None
):
    """
    运行动态量化实验
    
    Args:
        config_path: 配置文件路径
        results_dir: 结果保存目录
        baseline_model_path: 基线模型路径
        device: 设备
    """
    logger.info("="*80)
    logger.info("动态量化实验")
    logger.info("="*80)
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if device is None:
        device = config.get('device', 'cpu')
    
    logger.info(f"使用设备: {device}")
    
    # 创建结果目录
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # 加载基线模型
    logger.info(f"加载基线模型: {baseline_model_path}")
    
    if not Path(baseline_model_path).exists():
        logger.warning("基线模型不存在，创建新的 BERT 模型")
        bert_model = BERTModel(model_name="bert-base-uncased", num_labels=2)
    else:
        bert_model = BERTModel.load(baseline_model_path, num_labels=2)
    
    original_model = bert_model.get_model()
    original_model.to(device)
    original_model.eval()
    
    logger.info("✓ 基线模型加载成功")
    
    # 加载测试数据
    logger.info("加载测试数据...")
    
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
        logger.info("使用模拟数据...")
        # 创建模拟数据
        from torch.utils.data import DataLoader
        
        class DummyDataset:
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, 1000, (128,)),
                    'labels': torch.randint(0, 2, (1,)).item(),
                    'attention_mask': torch.ones(128)
                }
        
        test_dataset = DummyDataset()
        test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 评估原始模型
    logger.info("\n" + "="*80)
    logger.info("评估原始模型...")
    logger.info("="*80)
    
    original_size = calculate_model_size(original_model)
    logger.info(f"原始模型大小: {original_size:.2f} MB")
    
    original_metrics = evaluate_accuracy(original_model, test_loader, device, desc="原始模型")
    logger.info(f"原始模型准确率: {original_metrics['accuracy']:.4f}")
    
    original_timing = measure_inference_time(
        original_model, test_loader, device, 
        num_warmup=10, num_iterations=50
    )
    logger.info(f"原始模型推理时间: {original_timing['mean_ms']:.2f} ms")
    
    # 应用动态量化
    logger.info("\n" + "="*80)
    logger.info("应用动态量化...")
    logger.info("="*80)
    
    quantized_model = apply_dynamic_quantization(
        original_model,
        dtype=torch.qint8
    )
    
    logger.info("✓ 动态量化完成")
    
    # 评估量化模型
    logger.info("\n" + "="*80)
    logger.info("评估量化模型...")
    logger.info("="*80)
    
    quantized_size = calculate_model_size(quantized_model)
    logger.info(f"量化模型大小: {quantized_size:.2f} MB")
    
    quantized_metrics = evaluate_accuracy(quantized_model, test_loader, device, desc="量化模型")
    logger.info(f"量化模型准确率: {quantized_metrics['accuracy']:.4f}")
    
    quantized_timing = measure_inference_time(
        quantized_model, test_loader, device,
        num_warmup=10, num_iterations=50
    )
    logger.info(f"量化模型推理时间: {quantized_timing['mean_ms']:.2f} ms")
    
    # 对比分析
    logger.info("\n" + "="*80)
    logger.info("对比分析")
    logger.info("="*80)
    
    size_comparison = compare_model_sizes(original_model, quantized_model)
    
    accuracy_drop = original_metrics['accuracy'] - quantized_metrics['accuracy']
    speedup = original_timing['mean_ms'] / quantized_timing['mean_ms'] if quantized_timing['mean_ms'] > 0 else 0
    
    logger.info(f"\n准确率下降: {accuracy_drop:.4f} ({accuracy_drop*100:.2f}%)")
    logger.info(f"模型压缩比: {size_comparison['compression_ratio']:.2f}x")
    logger.info(f"推理加速比: {speedup:.2f}x")
    
    # 保存结果
    logger.info("\n" + "="*80)
    logger.info("保存结果...")
    logger.info("="*80)
    
    # 保存量化模型
    model_save_path = results_path / "quantized_model.pt"
    save_quantized_model(quantized_model, str(model_save_path))
    logger.info(f"✓ 模型已保存: {model_save_path}")
    
    # 保存实验结果
    import json
    results = {
        'original': {
            'accuracy': float(original_metrics['accuracy']),
            'f1_score': float(original_metrics['f1_score']),
            'size_mb': float(original_size),
            'inference_time_ms': float(original_timing['mean_ms'])
        },
        'quantized': {
            'accuracy': float(quantized_metrics['accuracy']),
            'f1_score': float(quantized_metrics['f1_score']),
            'size_mb': float(quantized_size),
            'inference_time_ms': float(quantized_timing['mean_ms'])
        },
        'comparison': {
            'accuracy_drop': float(accuracy_drop),
            'compression_ratio': float(size_comparison['compression_ratio']),
            'speedup': float(speedup)
        }
    }
    
    results_file = results_path / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 结果已保存: {results_file}")
    
    logger.info("\n" + "="*80)
    logger.info("动态量化实验完成!")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="动态量化实验")
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', help='配置文件')
    parser.add_argument('--results-dir', type=str, default='results/dynamic_quant', help='结果目录')
    parser.add_argument('--baseline-model', type=str, default='results/baseline/model', help='基线模型路径')
    parser.add_argument('--device', type=str, default=None, help='设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    run_dynamic_quantization_experiment(
        config_path=args.config,
        results_dir=args.results_dir,
        baseline_model_path=args.baseline_model,
        device=args.device
    )


if __name__ == "__main__":
    main()
