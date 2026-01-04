"""
剪枝实验

测试不同剪枝方法和稀疏度的效果
"""

import os
import sys
import torch
import yaml
import argparse
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_wrapper import BERTWrapper
from src.utils.data_loader import DataLoader
from src.training.evaluator import Evaluator
from src.pruning.structured_pruning import StructuredPruning
from src.pruning.unstructured_pruning import UnstructuredPruning
from src.pruning.progressive_pruning import ProgressivePruning
from src.utils.metrics import ModelSizeCalculator, InferenceTimer, calculate_sparsity


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def experiment_structured_pruning(model, config, data_loader, evaluator, device):
    """结构化剪枝实验"""
    print("\n" + "="*50)
    print("实验 1: 结构化剪枝")
    print("="*50)
    
    results = []
    
    # 获取校准数据
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # 测试不同数量的头剪枝
    num_heads_list = config['pruning']['attention_head_pruning'].get('num_heads_to_prune', [0, 4, 8, 12])
    if not isinstance(num_heads_list, list):
        num_heads_list = [num_heads_list]
    
    for num_heads in num_heads_list:
        # 重新加载模型
        model_copy = BERTWrapper.from_pretrained(
            config['checkpoint_dir'] + '/baseline_model',
            num_labels=config.get('num_labels', 2)
        ).to(device)
        
        print(f"\n--- 剪枝 {num_heads} 个注意力头 ---")
        
        # 应用剪枝
        pruner = StructuredPruning(model_copy, config['pruning'])
        pruner.apply_pruning(
            dataloader=train_loader,
            prune_heads=True,
            num_heads_to_prune=num_heads
        )
        
        # 评估
        eval_results = evaluator.evaluate(test_loader)
        
        # 计算模型大小
        size_info = ModelSizeCalculator.get_model_size(model_copy)
        
        result = {
            'method': 'structured_pruning',
            'num_heads_pruned': num_heads,
            'accuracy': eval_results['accuracy'],
            'f1': eval_results['f1'],
            'model_size_mb': size_info['size_mb']
        }
        
        results.append(result)
        
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")
        print(f"模型大小: {result['model_size_mb']:.2f} MB")
    
    return results


def experiment_unstructured_pruning(model, config, data_loader, evaluator, device):
    """非结构化剪枝实验"""
    print("\n" + "="*50)
    print("实验 2: 非结构化剪枝")
    print("="*50)
    
    results = []
    test_loader = data_loader.get_test_loader()
    
    # 测试不同稀疏度
    sparsity_levels = config['pruning']['unstructured'].get('sparsity_levels', [0.1, 0.3, 0.5, 0.7, 0.9])
    
    for sparsity in sparsity_levels:
        # 重新加载模型
        model_copy = BERTWrapper.from_pretrained(
            config['checkpoint_dir'] + '/baseline_model',
            num_labels=config.get('num_labels', 2)
        ).to(device)
        
        print(f"\n--- 稀疏度: {sparsity:.1%} ---")
        
        # 应用剪枝
        pruner = UnstructuredPruning(model_copy, config['pruning']['unstructured'])
        pruner.apply_pruning(sparsity=sparsity, global_pruning=True)
        
        # 评估
        eval_results = evaluator.evaluate(test_loader)
        
        # 计算实际稀疏度
        sparsity_info = calculate_sparsity(model_copy)
        
        # 计算模型大小
        size_info = ModelSizeCalculator.get_model_size(model_copy)
        
        result = {
            'method': 'unstructured_pruning',
            'target_sparsity': sparsity,
            'actual_sparsity': sparsity_info['overall_sparsity'],
            'accuracy': eval_results['accuracy'],
            'f1': eval_results['f1'],
            'model_size_mb': size_info['size_mb']
        }
        
        results.append(result)
        
        print(f"实际稀疏度: {result['actual_sparsity']:.4f}")
        print(f"准确率: {result['accuracy']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")
    
    return results


def experiment_progressive_pruning(model, config, data_loader, evaluator, device):
    """渐进式剪枝实验"""
    print("\n" + "="*50)
    print("实验 3: 渐进式剪枝")
    print("="*50)
    
    # 重新加载模型
    model = BERTWrapper.from_pretrained(
        config['checkpoint_dir'] + '/baseline_model',
        num_labels=config.get('num_labels', 2)
    ).to(device)
    
    test_loader = data_loader.get_test_loader()
    
    # 应用渐进式剪枝
    progressive_config = config['pruning'].get('progressive', {})
    pruner = ProgressivePruning(model, progressive_config)
    
    results = []
    
    # 获取剪枝计划
    schedule = pruner.get_pruning_schedule()
    print(f"剪枝计划: {schedule}")
    
    # 逐步剪枝
    for iteration in range(progressive_config.get('num_iterations', 5)):
        stats = pruner.prune_iteration(iteration, global_pruning=True)
        
        # 评估
        eval_results = evaluator.evaluate(test_loader)
        
        result = {
            'method': 'progressive_pruning',
            'iteration': iteration,
            'sparsity': stats['actual_sparsity'],
            'accuracy': eval_results['accuracy'],
            'f1': eval_results['f1']
        }
        
        results.append(result)
        
        print(f"\n迭代 {iteration}: 稀疏度={result['sparsity']:.4f}, "
              f"准确率={result['accuracy']:.4f}")
    
    return results


def main(args):
    """主函数"""
    # 加载配置
    base_config = load_config('configs/base_config.yaml')
    pruning_config = load_config(args.config)
    
    # 合并配置
    config = {
        **base_config.get('model', {}),
        **base_config.get('data', {}),
        **base_config.get('output', {}),
        'pruning': pruning_config.get('pruning', {})
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = config.get('results_dir', './results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n=== 加载数据 ===")
    data_loader = DataLoader(config)
    data_loader.prepare_data(
        tokenizer_name=config.get('name', 'bert-base-uncased')
    )
    
    # 加载基线模型
    print("\n=== 加载基线模型 ===")
    model = BERTWrapper.from_pretrained(
        config['checkpoint_dir'] + '/baseline_model',
        num_labels=config.get('num_labels', 2)
    ).to(device)
    
    evaluator = Evaluator(model, device)
    
    # 运行实验
    all_results = []
    
    if args.structured:
        results = experiment_structured_pruning(model, config, data_loader, evaluator, device)
        all_results.extend(results)
    
    if args.unstructured:
        results = experiment_unstructured_pruning(model, config, data_loader, evaluator, device)
        all_results.extend(results)
    
    if args.progressive:
        results = experiment_progressive_pruning(model, config, data_loader, evaluator, device)
        all_results.extend(results)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'pruning_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n所有结果已保存到: {results_path}")
    
    # 打印总结
    print("\n=== 实验总结 ===")
    for result in all_results:
        print(f"{result['method']}: 准确率={result['accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='剪枝实验')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pruning_config.yaml',
        help='剪枝配置文件路径'
    )
    parser.add_argument(
        '--structured',
        action='store_true',
        help='运行结构化剪枝实验'
    )
    parser.add_argument(
        '--unstructured',
        action='store_true',
        help='运行非结构化剪枝实验'
    )
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='运行渐进式剪枝实验'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何实验，运行所有实验
    if not (args.structured or args.unstructured or args.progressive):
        args.structured = True
        args.unstructured = True
        args.progressive = True
    
    main(args)
