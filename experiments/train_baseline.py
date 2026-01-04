"""
训练基线模型

训练一个基线BERT模型用于后续剪枝和量化实验
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_wrapper import BERTWrapper
from src.utils.data_loader import DataLoader
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.metrics import ModelSizeCalculator


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(args):
    """主函数"""
    # 加载配置
    base_config = load_config(args.config)
    
    # 合并配置
    config = {
        **base_config.get('model', {}),
        **base_config.get('training', {}),
        **base_config.get('data', {}),
        **base_config.get('output', {})
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = config.get('output_dir', './outputs')
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化数据加载器
    print("\n=== 加载数据 ===")
    data_loader = DataLoader(config)
    data_loader.prepare_data(
        tokenizer_name=config.get('name', 'bert-base-uncased')
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # 初始化模型
    print("\n=== 初始化模型 ===")
    model = BERTWrapper(
        model_name=config.get('name', 'bert-base-uncased'),
        num_labels=config.get('num_labels', 2),
        config={
            'hidden_dropout_prob': config.get('hidden_dropout_prob', 0.1),
            'attention_probs_dropout_prob': config.get('attention_probs_dropout_prob', 0.1)
        }
    )
    
    # 打印模型信息
    num_params = model.get_num_parameters()
    model_size = model.get_model_size()
    print(f"模型参数数量: {num_params:,}")
    print(f"模型大小: {model_size:.2f} MB")
    
    # 初始化训练器和评估器
    trainer = Trainer(model, config, device)
    evaluator = Evaluator(model, device)
    
    # 训练模型
    print("\n=== 开始训练 ===")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator
    )
    
    # 保存模型
    model_save_path = os.path.join(checkpoint_dir, 'baseline_model')
    print(f"\n保存模型到: {model_save_path}")
    model.save_pretrained(model_save_path)
    
    # 保存训练历史
    import json
    history_path = os.path.join(output_dir, 'baseline_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    # 最终评估
    print("\n=== 最终评估 ===")
    test_metrics = evaluator.evaluate(test_loader)
    
    # 保存评估结果
    results = {
        'model_name': 'BERT Baseline',
        'num_parameters': num_params,
        'model_size_mb': model_size,
        **test_metrics
    }
    
    results_path = os.path.join(output_dir, 'baseline_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {results_path}")
    
    print("\n=== 基线模型训练完成 ===")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练基线BERT模型')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    main(args)
