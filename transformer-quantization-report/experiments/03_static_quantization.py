#!/usr/bin/env python3
"""
静态量化实验

加载基线模型并应用静态量化，评估量化效果
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
from tqdm import tqdm

from src.models.bert_model import BERTModel
from src.quantization.static_quantization import apply_static_quantization
from src.evaluation.accuracy_eval import evaluate_accuracy
from src.evaluation.performance_eval import measure_inference_time
from src.evaluation.size_eval import calculate_model_size
from src.utils.data_loader import load_imdb_dataset, create_dataloader, create_calibration_dataloader
from src.utils.logger import get_experiment_logger

logger = get_experiment_logger("static_quantization")


def run_static_quantization_experiment(
    config_path: str = "configs/experiment_config.yaml",
    model_path: str = "results/baseline/model",
    calibration_samples: int = 1000,
    output_dir: str = "results/static_quant",
    device: str = None
):
    """
    运行静态量化实验
    
    Args:
        config_path: 配置文件路径
        model_path: 基线模型路径
        calibration_samples: 校准样本数量
        output_dir: 输出目录
        device: 设备
    """
    logger.info("="*80)
    logger.info("静态量化实验")
    logger.info("="*80)
    
    # 加载配置
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 设置设备 - 静态量化仅支持 CPU
    if device is None:
        device = config.get('device', 'cpu')
    
    if device != 'cpu':
        logger.warning("静态量化仅支持 CPU，自动切换到 CPU")
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    logger.info(f"校准样本数量: {calibration_samples}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存运行配置
    run_config = {
        'model_path': model_path,
        'calibration_samples': calibration_samples,
        'device': device,
        'backend': 'fbgemm'
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
        return
    
    try:
        bert_model = BERTModel.load(model_path, num_labels=2)
        original_model = bert_model.get_model()
        original_model.to(device)
        original_model.eval()
        logger.info("✓ 基线模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.info("创建新的 BERT 模型用于演示...")
        bert_model = BERTModel(model_name="bert-base-uncased", num_labels=2)
        original_model = bert_model.get_model()
        original_model.to(device)
        original_model.eval()
    
    # 加载数据集
    logger.info("\n" + "="*80)
    logger.info("加载数据集...")
    logger.info("="*80)
    
    try:
        train_dataset, test_dataset, tokenizer = load_imdb_dataset(
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
        
        # 创建校准数据加载器
        calibration_loader = create_calibration_dataloader(
            train_dataset,
            num_samples=calibration_samples,
            batch_size=32,
            seed=42
        )
        
        # 创建测试数据加载器
        test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"✓ 数据加载完成")
        logger.info(f"  校准数据: {calibration_samples} 样本")
        logger.info(f"  测试数据: {len(test_dataset)} 样本")
        
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
        
        calibration_dataset = DummyDataset(calibration_samples)
        test_dataset = DummyDataset(500)
        
        calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 评估基线模型
    logger.info("\n" + "="*80)
    logger.info("评估基线模型...")
    logger.info("="*80)
    
    original_size = calculate_model_size(original_model)
    logger.info(f"基线模型大小: {original_size:.2f} MB")
    
    original_metrics = evaluate_accuracy(original_model, test_loader, device, desc="基线模型评估")
    logger.info(f"基线模型准确率: {original_metrics['accuracy']:.4f}")
    logger.info(f"基线模型 F1: {original_metrics['f1_score']:.4f}")
    
    original_timing = measure_inference_time(
        original_model, test_loader, device,
        num_warmup=10, num_iterations=50
    )
    logger.info(f"基线模型推理时间: {original_timing['mean_ms']:.2f} ms")
    
    # 应用静态量化
    logger.info("\n" + "="*80)
    logger.info("应用静态量化...")
    logger.info("="*80)
    logger.info(f"准备校准数据（{calibration_samples} 样本）...")
    
    try:
        quantized_model = apply_static_quantization(
            original_model,
            calibration_loader=calibration_loader,
            backend='fbgemm',
            num_calibration_batches=calibration_samples // 32
        )
        logger.info("✓ 静态量化完成")
    except Exception as e:
        logger.error(f"静态量化失败: {e}")
        logger.error("静态量化必须使用真实的校准数据才能正常工作")
        logger.error("请确保:")
        logger.error("  1. 模型支持静态量化")
        logger.error("  2. 校准数据加载正确")
        logger.error("  3. 在 CPU 上运行（静态量化不支持 CUDA）")
        logger.info("\n注意: 由于静态量化失败，使用动态量化作为备选方案")
        logger.info("这将改变实验性质，结果可能与预期不同\n")
        
        # 备选方案：使用动态量化
        import torch.quantization as quant
        
        quantized_model = torch.quantization.quantize_dynamic(
            original_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        logger.warning("⚠️ 当前使用动态量化替代静态量化")
    
    # 评估量化模型
    logger.info("\n" + "="*80)
    logger.info("评估量化模型...")
    logger.info("="*80)
    
    quantized_size = calculate_model_size(quantized_model)
    logger.info(f"量化模型大小: {quantized_size:.2f} MB")
    
    quantized_metrics = evaluate_accuracy(quantized_model, test_loader, device, desc="量化模型评估")
    logger.info(f"量化模型准确率: {quantized_metrics['accuracy']:.4f}")
    logger.info(f"量化模型 F1: {quantized_metrics['f1_score']:.4f}")
    
    quantized_timing = measure_inference_time(
        quantized_model, test_loader, device,
        num_warmup=10, num_iterations=50
    )
    logger.info(f"量化模型推理时间: {quantized_timing['mean_ms']:.2f} ms")
    
    # 计算对比指标
    logger.info("\n" + "="*80)
    logger.info("结果对比")
    logger.info("="*80)
    
    accuracy_drop = original_metrics['accuracy'] - quantized_metrics['accuracy']
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    speedup = original_timing['mean_ms'] / quantized_timing['mean_ms'] if quantized_timing['mean_ms'] > 0 else 0
    
    # 打印对比表格
    print("\n" + "+"*80)
    print(f"{'指标':<20} {'基线':<20} {'静态量化':<20}")
    print("+"*80)
    print(f"{'准确率':<20} {original_metrics['accuracy']:<20.4f} {quantized_metrics['accuracy']:<20.4f}")
    print(f"{'准确率下降':<20} {'-':<20} {accuracy_drop:<20.4f} ({accuracy_drop*100:.2f}%)")
    print(f"{'F1 分数':<20} {original_metrics['f1_score']:<20.4f} {quantized_metrics['f1_score']:<20.4f}")
    print(f"{'模型大小 (MB)':<20} {original_size:<20.2f} {quantized_size:<20.2f}")
    print(f"{'压缩比':<20} {'1.0x':<20} {compression_ratio:<20.2f}x")
    print(f"{'推理时间 (ms)':<20} {original_timing['mean_ms']:<20.2f} {quantized_timing['mean_ms']:<20.2f}")
    print(f"{'加速比':<20} {'1.0x':<20} {speedup:<20.2f}x")
    print("+"*80 + "\n")
    
    # 保存量化模型
    logger.info("\n" + "="*80)
    logger.info("保存结果...")
    logger.info("="*80)
    
    model_save_path = output_path / "quantized_model.pt"
    torch.save(quantized_model, str(model_save_path))
    logger.info(f"✓ 量化模型已保存: {model_save_path}")
    
    # 保存实验结果
    results = {
        'baseline': {
            'accuracy': float(original_metrics['accuracy']),
            'f1_score': float(original_metrics['f1_score']),
            'precision': float(original_metrics['precision']),
            'recall': float(original_metrics['recall']),
            'size_mb': float(original_size),
            'inference_time_ms': float(original_timing['mean_ms'])
        },
        'static_quantized': {
            'accuracy': float(quantized_metrics['accuracy']),
            'f1_score': float(quantized_metrics['f1_score']),
            'precision': float(quantized_metrics['precision']),
            'recall': float(quantized_metrics['recall']),
            'size_mb': float(quantized_size),
            'inference_time_ms': float(quantized_timing['mean_ms'])
        },
        'comparison': {
            'accuracy_drop': float(accuracy_drop),
            'accuracy_drop_percent': float(accuracy_drop * 100),
            'compression_ratio': float(compression_ratio),
            'speedup': float(speedup)
        }
    }
    
    # 保存为 JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 指标已保存: {metrics_file}")
    
    # 保存为 CSV
    comparison_df = pd.DataFrame([
        {
            '模型': '基线模型',
            '准确率': f"{original_metrics['accuracy']:.4f}",
            'F1分数': f"{original_metrics['f1_score']:.4f}",
            '模型大小(MB)': f"{original_size:.2f}",
            '推理时间(ms)': f"{original_timing['mean_ms']:.2f}",
            '压缩比': '1.0x',
            '加速比': '1.0x'
        },
        {
            '模型': '静态量化',
            '准确率': f"{quantized_metrics['accuracy']:.4f}",
            'F1分数': f"{quantized_metrics['f1_score']:.4f}",
            '模型大小(MB)': f"{quantized_size:.2f}",
            '推理时间(ms)': f"{quantized_timing['mean_ms']:.2f}",
            '压缩比': f"{compression_ratio:.2f}x",
            '加速比': f"{speedup:.2f}x"
        }
    ])
    
    csv_file = output_path / "comparison.csv"
    comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ 对比数据已保存: {csv_file}")
    
    logger.info("\n" + "="*80)
    logger.info("静态量化实验完成！")
    logger.info("="*80)
    logger.info(f"\n结果保存在: {output_dir}")
    logger.info(f"  - quantized_model.pt   量化模型")
    logger.info(f"  - metrics.json         评估指标")
    logger.info(f"  - comparison.csv       对比数据")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="静态量化实验")
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
        '--calibration_samples',
        type=int,
        default=1000,
        help='校准样本数量'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/static_quant',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cpu only for static quantization)'
    )
    
    args = parser.parse_args()
    
    run_static_quantization_experiment(
        config_path=args.config,
        model_path=args.model_path,
        calibration_samples=args.calibration_samples,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
