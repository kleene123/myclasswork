#!/usr/bin/env python3
"""
基线模型训练

训练基线 BERT 或 DistilBERT 模型，用于文本分类任务
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
import yaml
import argparse
import json
import numpy as np
from tqdm import tqdm

from src.models.bert_model import BERTModel
from src.utils.data_loader import load_imdb_dataset, create_dataloader
from src.evaluation.accuracy_eval import evaluate_accuracy
from src.evaluation.performance_eval import measure_inference_time
from src.evaluation.size_eval import calculate_model_size
from src.utils.logger import get_experiment_logger
from src.utils.visualization import plot_training_curves

logger = get_experiment_logger("baseline_training")


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    device: str,
    epoch: int,
    max_grad_norm: float = 1.0
):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epoch: 当前 epoch
        max_grad_norm: 最大梯度范数
        
    Returns:
        avg_loss: 平均损失
        avg_accuracy: 平均准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # 准备数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # 优化器更新
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = correct / total
    
    return avg_loss, avg_accuracy


def validate(model: nn.Module, val_loader, device: str):
    """
    验证模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备
        
    Returns:
        avg_loss: 平均损失
        metrics: 评估指标
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    
    # 计算准确率等指标
    metrics = evaluate_accuracy(model, val_loader, device, desc="计算验证指标")
    
    return avg_loss, metrics


def run_baseline_training(
    config_path: str = "configs/training_config.yaml",
    model_name: str = "bert-base-uncased",
    dataset: str = "imdb",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    output_dir: str = "results/baseline",
    device: str = None
):
    """
    运行基线模型训练
    
    Args:
        config_path: 配置文件路径
        model_name: 模型名称
        dataset: 数据集名称
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        output_dir: 输出目录
        device: 设备
    """
    logger.info("="*80)
    logger.info("基线模型训练")
    logger.info("="*80)
    
    # 设置随机种子
    set_seed(42)
    logger.info("✓ 随机种子已设置为 42")
    
    # 加载配置
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 从配置文件读取参数（如果未通过命令行指定）
    training_config = config.get('training', {}).get('baseline', {})
    if num_epochs == 3 and 'num_epochs' in training_config:
        num_epochs = training_config['num_epochs']
    if batch_size == 32 and 'batch_size' in training_config:
        batch_size = training_config['batch_size']
    if learning_rate == 2e-5 and 'learning_rate' in training_config:
        learning_rate = training_config['learning_rate']
    
    # 设置设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"使用设备: {device}")
    logger.info(f"模型: {model_name}")
    logger.info(f"数据集: {dataset}")
    logger.info(f"训练轮数: {num_epochs}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"学习率: {learning_rate}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存运行配置
    run_config = {
        'model_name': model_name,
        'dataset': dataset,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': device,
        'seed': 42
    }
    
    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    
    # 加载数据集
    logger.info("\n" + "="*80)
    logger.info("加载数据集...")
    logger.info("="*80)
    
    try:
        train_dataset, test_dataset, tokenizer = load_imdb_dataset(
            tokenizer_name=model_name,
            max_length=128
        )
        
        # 创建验证集（从训练集中分割）
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据加载器
        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"✓ 数据加载完成")
        logger.info(f"  训练集: {len(train_dataset)} 样本")
        logger.info(f"  验证集: {len(val_dataset)} 样本")
        logger.info(f"  测试集: {len(test_dataset)} 样本")
        
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
        
        train_dataset = DummyDataset(1000)
        val_dataset = DummyDataset(200)
        test_dataset = DummyDataset(500)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    logger.info("\n" + "="*80)
    logger.info("初始化模型...")
    logger.info("="*80)
    
    bert_model = BERTModel(model_name=model_name, num_labels=2)
    model = bert_model.get_model()
    model.to(device)
    
    logger.info("✓ 模型初始化完成")
    
    # 计算模型大小
    model_size = calculate_model_size(model)
    logger.info(f"模型大小: {model_size:.2f} MB")
    
    # 设置优化器和调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = training_config.get('warmup_steps', 500)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"优化器: AdamW")
    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"预热步数: {warmup_steps}")
    
    # 训练循环
    logger.info("\n" + "="*80)
    logger.info("开始训练...")
    logger.info("="*80)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0
    
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 80)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, max_grad_norm
        )
        
        # 验证
        val_loss, val_metrics = validate(model, val_loader, device)
        val_acc = val_metrics['accuracy']
        
        # 记录
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f"\nEpoch {epoch} 结果:")
        logger.info(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        logger.info(f"  验证 F1: {val_metrics['f1_score']:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            logger.info(f"  ✓ 新的最佳模型！验证准确率: {best_val_accuracy:.4f}")
            bert_model.save(str(output_path / "model"))
            logger.info(f"  ✓ 模型已保存")
    
    logger.info("\n" + "="*80)
    logger.info("训练完成！")
    logger.info("="*80)
    
    # 在测试集上评估
    logger.info("\n" + "="*80)
    logger.info("在测试集上评估...")
    logger.info("="*80)
    
    # 加载最佳模型
    best_model = BERTModel.load(str(output_path / "model"), num_labels=2)
    best_model.to(device)
    best_model.eval()
    model = best_model.get_model()
    
    # 评估准确率
    test_metrics = evaluate_accuracy(model, test_loader, device, desc="测试集评估")
    
    # 评估推理时间
    inference_timing = measure_inference_time(
        model, test_loader, device,
        num_warmup=10, num_iterations=100
    )
    
    # 模型大小
    model_size = calculate_model_size(model)
    
    logger.info(f"\n测试集结果:")
    logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 分数: {test_metrics['f1_score']:.4f}")
    logger.info(f"  精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  推理时间: {inference_timing['mean_ms']:.2f} ms")
    logger.info(f"  模型大小: {model_size:.2f} MB")
    
    # 保存评估结果
    results = {
        'test_metrics': {
            'accuracy': float(test_metrics['accuracy']),
            'f1_score': float(test_metrics['f1_score']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall'])
        },
        'performance': {
            'inference_time_ms': float(inference_timing['mean_ms']),
            'model_size_mb': float(model_size)
        },
        'training_history': {
            'train_losses': [float(x) for x in train_losses],
            'train_accuracies': [float(x) for x in train_accuracies],
            'val_losses': [float(x) for x in val_losses],
            'val_accuracies': [float(x) for x in val_accuracies]
        }
    }
    
    # 保存为 JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ 指标已保存: {metrics_file}")
    
    # 保存为 CSV
    import pandas as pd
    
    eval_df = pd.DataFrame([{
        '指标': '准确率',
        '值': f"{test_metrics['accuracy']:.4f}"
    }, {
        '指标': 'F1 分数',
        '值': f"{test_metrics['f1_score']:.4f}"
    }, {
        '指标': '推理时间 (ms)',
        '值': f"{inference_timing['mean_ms']:.2f}"
    }, {
        '指标': '模型大小 (MB)',
        '值': f"{model_size:.2f}"
    }])
    
    csv_file = output_path / "evaluation_results.csv"
    eval_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logger.info(f"✓ 评估结果已保存: {csv_file}")
    
    # 生成训练曲线图
    logger.info("\n生成训练曲线图...")
    try:
        plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            save_path=str(output_path / "training_curve.png")
        )
        logger.info(f"✓ 训练曲线已保存: {output_path / 'training_curve.png'}")
    except Exception as e:
        logger.warning(f"生成训练曲线失败: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("基线模型训练完成！")
    logger.info("="*80)
    logger.info(f"\n结果保存在: {output_dir}")
    logger.info(f"  - model/               训练好的模型")
    logger.info(f"  - metrics.json         评估指标")
    logger.info(f"  - evaluation_results.csv  评估结果表")
    logger.info(f"  - training_curve.png   训练曲线图")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="基线模型训练")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert-base-uncased',
        help='模型名称'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='imdb',
        help='数据集名称'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='学习率'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baseline',
        help='输出目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    run_baseline_training(
        config_path=args.config,
        model_name=args.model,
        dataset=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
