"""
量化实验

测试不同量化方法的效果
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
from src.quantization.ptq import PostTrainingQuantization
from src.quantization.qat import QuantizationAwareTraining
from src.utils.metrics import ModelSizeCalculator, InferenceTimer


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def experiment_dynamic_quantization(model, config, data_loader, evaluator, device):
    """动态量化实验"""
    print("\n" + "="*50)
    print("实验 1: 动态量化")
    print("="*50)
    
    # 复制模型
    model_copy = copy.deepcopy(model)
    
    # 应用动态量化
    ptq = PostTrainingQuantization(model_copy, config['quantization']['ptq'])
    quantized_model = ptq.apply_dynamic_quantization(dtype=torch.qint8)
    
    # 评估
    test_loader = data_loader.get_test_loader()
    evaluator_quant = Evaluator(quantized_model, device)
    eval_results = evaluator_quant.evaluate(test_loader)
    
    # 计算模型大小
    size_info = ModelSizeCalculator.get_model_size(quantized_model)
    original_size = ModelSizeCalculator.get_model_size(model)
    
    result = {
        'method': 'dynamic_quantization',
        'dtype': 'int8',
        'accuracy': eval_results['accuracy'],
        'f1': eval_results['f1'],
        'model_size_mb': size_info['size_mb'],
        'original_size_mb': original_size['size_mb'],
        'compression_ratio': original_size['size_mb'] / size_info['size_mb'] if size_info['size_mb'] > 0 else 0
    }
    
    print(f"\n准确率: {result['accuracy']:.4f}")
    print(f"F1分数: {result['f1']:.4f}")
    print(f"模型大小: {result['model_size_mb']:.2f} MB")
    print(f"压缩比: {result['compression_ratio']:.2f}x")
    
    return result


def experiment_static_quantization(model, config, data_loader, evaluator, device):
    """静态量化实验"""
    print("\n" + "="*50)
    print("实验 2: 静态量化")
    print("="*50)
    
    # 复制模型
    model_copy = copy.deepcopy(model)
    
    # 准备校准数据
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # 应用静态量化
    try:
        ptq = PostTrainingQuantization(model_copy, config['quantization']['ptq'])
        quantized_model = ptq.apply_static_quantization(train_loader)
        
        # 评估
        evaluator_quant = Evaluator(quantized_model, device)
        eval_results = evaluator_quant.evaluate(test_loader)
        
        # 计算模型大小
        size_info = ModelSizeCalculator.get_model_size(quantized_model)
        original_size = ModelSizeCalculator.get_model_size(model)
        
        result = {
            'method': 'static_quantization',
            'dtype': 'int8',
            'accuracy': eval_results['accuracy'],
            'f1': eval_results['f1'],
            'model_size_mb': size_info['size_mb'],
            'original_size_mb': original_size['size_mb'],
            'compression_ratio': original_size['size_mb'] / size_info['size_mb'] if size_info['size_mb'] > 0 else 0
        }
        
        print(f"\n准确率: {result['accuracy']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")
        print(f"模型大小: {result['model_size_mb']:.2f} MB")
        print(f"压缩比: {result['compression_ratio']:.2f}x")
        
    except Exception as e:
        print(f"静态量化失败: {e}")
        print("注意: BERT模型的静态量化可能需要特殊处理")
        result = {
            'method': 'static_quantization',
            'error': str(e)
        }
    
    return result


def experiment_qat(model, config, data_loader, evaluator, device):
    """量化感知训练实验"""
    print("\n" + "="*50)
    print("实验 3: 量化感知训练 (QAT)")
    print("="*50)
    
    # 复制模型
    model_copy = copy.deepcopy(model)
    
    try:
        # 准备数据
        train_loader = data_loader.get_train_loader()
        test_loader = data_loader.get_test_loader()
        
        # 设置QAT
        qat = QuantizationAwareTraining(model_copy, config['quantization']['qat'])
        
        # 准备优化器和损失函数
        optimizer = torch.optim.AdamW(
            model_copy.parameters(),
            lr=config['quantization']['qat'].get('learning_rate', 1e-5)
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # 应用QAT
        quantized_model, history = qat.apply_qat(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config['quantization']['qat'].get('epochs', 2),
            device=device
        )
        
        # 评估
        evaluator_quant = Evaluator(quantized_model, device)
        eval_results = evaluator_quant.evaluate(test_loader)
        
        # 计算模型大小
        size_info = ModelSizeCalculator.get_model_size(quantized_model)
        original_size = ModelSizeCalculator.get_model_size(model)
        
        result = {
            'method': 'quantization_aware_training',
            'dtype': 'int8',
            'accuracy': eval_results['accuracy'],
            'f1': eval_results['f1'],
            'model_size_mb': size_info['size_mb'],
            'original_size_mb': original_size['size_mb'],
            'compression_ratio': original_size['size_mb'] / size_info['size_mb'] if size_info['size_mb'] > 0 else 0
        }
        
        print(f"\n准确率: {result['accuracy']:.4f}")
        print(f"F1分数: {result['f1']:.4f}")
        print(f"模型大小: {result['model_size_mb']:.2f} MB")
        print(f"压缩比: {result['compression_ratio']:.2f}x")
        
    except Exception as e:
        print(f"QAT失败: {e}")
        print("注意: BERT模型的QAT可能需要特殊处理")
        result = {
            'method': 'quantization_aware_training',
            'error': str(e)
        }
    
    return result


def main(args):
    """主函数"""
    # 加载配置
    base_config = load_config('configs/base_config.yaml')
    quant_config = load_config(args.config)
    
    # 合并配置
    config = {
        **base_config.get('model', {}),
        **base_config.get('data', {}),
        **base_config.get('output', {}),
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
    
    if args.dynamic:
        result = experiment_dynamic_quantization(model, config, data_loader, evaluator, device)
        all_results.append(result)
    
    if args.static:
        result = experiment_static_quantization(model, config, data_loader, evaluator, device)
        all_results.append(result)
    
    if args.qat:
        result = experiment_qat(model, config, data_loader, evaluator, device)
        all_results.append(result)
    
    # 保存结果
    results_path = os.path.join(output_dir, 'quantization_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n所有结果已保存到: {results_path}")
    
    # 打印总结
    print("\n=== 实验总结 ===")
    for result in all_results:
        if 'error' not in result:
            print(f"{result['method']}: 准确率={result['accuracy']:.4f}, "
                  f"压缩比={result.get('compression_ratio', 0):.2f}x")
        else:
            print(f"{result['method']}: 失败 - {result['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='量化实验')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/quantization_config.yaml',
        help='量化配置文件路径'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='运行动态量化实验'
    )
    parser.add_argument(
        '--static',
        action='store_true',
        help='运行静态量化实验'
    )
    parser.add_argument(
        '--qat',
        action='store_true',
        help='运行量化感知训练实验'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何实验，运行所有实验
    if not (args.dynamic or args.static or args.qat):
        args.dynamic = True
        args.static = True
        args.qat = True
    
    main(args)
