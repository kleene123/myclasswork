"""
剪枝+量化组合实验

测试先剪枝后量化和先量化后剪枝的效果对比
"""

import os
import sys
import torch
import yaml
import argparse
import json
import copy
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.bert_wrapper import BERTWrapper
from src.utils.data_loader import DataLoader
from src.training.evaluator import Evaluator
from src.pruning.unstructured_pruning import UnstructuredPruning
from src.quantization.ptq import PostTrainingQuantization
from src.utils.metrics import ModelSizeCalculator, calculate_sparsity


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def pruning_then_quantization(model, config, data_loader, evaluator, device):
    """先剪枝后量化"""
    print("\n" + "="*50)
    print("策略 1: 先剪枝后量化")
    print("="*50)
    
    # 复制模型
    model_copy = copy.deepcopy(model)
    
    # 步骤1: 剪枝
    print("\n步骤 1: 应用非结构化剪枝...")
    sparsity = config['pruning']['sparsity']
    pruner = UnstructuredPruning(model_copy, config['pruning']['unstructured'])
    pruner.apply_pruning(sparsity=sparsity, global_pruning=True)
    
    # 评估剪枝后的模型
    test_loader = data_loader.get_test_loader()
    eval_pruned = evaluator.evaluate(test_loader)
    sparsity_info = calculate_sparsity(model_copy)
    
    print(f"剪枝后准确率: {eval_pruned['accuracy']:.4f}")
    print(f"实际稀疏度: {sparsity_info['overall_sparsity']:.4f}")
    
    # 步骤2: 量化
    print("\n步骤 2: 应用动态量化...")
    ptq = PostTrainingQuantization(model_copy, config['quantization']['ptq'])
    quantized_model = ptq.apply_dynamic_quantization(dtype=torch.qint8)
    
    # 最终评估
    evaluator_final = Evaluator(quantized_model, device)
    eval_final = evaluator_final.evaluate(test_loader)
    
    # 计算模型大小
    size_info = ModelSizeCalculator.get_model_size(quantized_model)
    original_size = ModelSizeCalculator.get_model_size(model)
    
    result = {
        'strategy': 'pruning_then_quantization',
        'pruning_sparsity': sparsity_info['overall_sparsity'],
        'accuracy_after_pruning': eval_pruned['accuracy'],
        'accuracy_final': eval_final['accuracy'],
        'f1_final': eval_final['f1'],
        'model_size_mb': size_info['size_mb'],
        'original_size_mb': original_size['size_mb'],
        'compression_ratio': original_size['size_mb'] / size_info['size_mb'] if size_info['size_mb'] > 0 else 0
    }
    
    print(f"\n最终准确率: {result['accuracy_final']:.4f}")
    print(f"最终F1分数: {result['f1_final']:.4f}")
    print(f"模型大小: {result['model_size_mb']:.2f} MB")
    print(f"总压缩比: {result['compression_ratio']:.2f}x")
    
    return result


def quantization_then_pruning(model, config, data_loader, evaluator, device):
    """先量化后剪枝"""
    print("\n" + "="*50)
    print("策略 2: 先量化后剪枝")
    print("="*50)
    
    # 复制模型
    model_copy = copy.deepcopy(model)
    
    # 步骤1: 量化
    print("\n步骤 1: 应用动态量化...")
    ptq = PostTrainingQuantization(model_copy, config['quantization']['ptq'])
    quantized_model = ptq.apply_dynamic_quantization(dtype=torch.qint8)
    
    # 评估量化后的模型
    test_loader = data_loader.get_test_loader()
    evaluator_quant = Evaluator(quantized_model, device)
    eval_quant = evaluator_quant.evaluate(test_loader)
    
    print(f"量化后准确率: {eval_quant['accuracy']:.4f}")
    
    # 步骤2: 剪枝
    print("\n步骤 2: 应用非结构化剪枝...")
    print("注意: 量化后的模型剪枝可能效果不佳")
    
    try:
        sparsity = config['pruning']['sparsity']
        pruner = UnstructuredPruning(quantized_model, config['pruning']['unstructured'])
        pruner.apply_pruning(sparsity=sparsity, global_pruning=True)
        
        # 最终评估
        eval_final = evaluator_quant.evaluate(test_loader)
        sparsity_info = calculate_sparsity(quantized_model)
        
        # 计算模型大小
        size_info = ModelSizeCalculator.get_model_size(quantized_model)
        original_size = ModelSizeCalculator.get_model_size(model)
        
        result = {
            'strategy': 'quantization_then_pruning',
            'pruning_sparsity': sparsity_info['overall_sparsity'],
            'accuracy_after_quantization': eval_quant['accuracy'],
            'accuracy_final': eval_final['accuracy'],
            'f1_final': eval_final['f1'],
            'model_size_mb': size_info['size_mb'],
            'original_size_mb': original_size['size_mb'],
            'compression_ratio': original_size['size_mb'] / size_info['size_mb'] if size_info['size_mb'] > 0 else 0
        }
        
        print(f"\n最终准确率: {result['accuracy_final']:.4f}")
        print(f"最终F1分数: {result['f1_final']:.4f}")
        print(f"模型大小: {result['model_size_mb']:.2f} MB")
        print(f"总压缩比: {result['compression_ratio']:.2f}x")
        
    except Exception as e:
        print(f"量化后剪枝失败: {e}")
        result = {
            'strategy': 'quantization_then_pruning',
            'accuracy_after_quantization': eval_quant['accuracy'],
            'error': str(e)
        }
    
    return result


def main(args):
    """主函数"""
    # 加载配置
    base_config = load_config('configs/base_config.yaml')
    pruning_config = load_config('configs/pruning_config.yaml')
    quant_config = load_config('configs/quantization_config.yaml')
    
    # 合并配置
    config = {
        **base_config.get('model', {}),
        **base_config.get('data', {}),
        **base_config.get('output', {}),
        'pruning': pruning_config.get('pruning', {}),
        'quantization': quant_config.get('quantization', {})
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
    
    # 策略1: 先剪枝后量化
    result1 = pruning_then_quantization(model, config, data_loader, evaluator, device)
    all_results.append(result1)
    
    # 策略2: 先量化后剪枝
    result2 = quantization_then_pruning(model, config, data_loader, evaluator, device)
    all_results.append(result2)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'combined_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n所有结果已保存到: {results_path}")
    
    # 打印对比
    print("\n=== 策略对比 ===")
    for result in all_results:
        if 'error' not in result:
            print(f"\n{result['strategy']}:")
            print(f"  准确率: {result['accuracy_final']:.4f}")
            print(f"  压缩比: {result.get('compression_ratio', 0):.2f}x")
        else:
            print(f"\n{result['strategy']}: 失败")
    
    # 推荐最佳策略
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['accuracy_final'])
        print(f"\n推荐策略: {best_result['strategy']}")
        print(f"准确率: {best_result['accuracy_final']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='剪枝+量化组合实验')
    
    args = parser.parse_args()
    main(args)
