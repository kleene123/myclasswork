#!/usr/bin/env python3
"""
快速演示脚本

展示如何使用项目的核心功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.models.bert_model import BERTModel
from src.quantization.dynamic_quantization import apply_dynamic_quantization
from src.evaluation.size_eval import calculate_model_size


def demo():
    """快速演示"""
    
    print("="*80)
    print("Transformer 模型量化演示")
    print("="*80)
    
    print("\n1. 创建 BERT 模型...")
    bert_model = BERTModel(model_name="bert-base-uncased", num_labels=2)
    model = bert_model.get_model()
    model.eval()
    print("✓ 模型创建成功")
    
    print("\n2. 计算原始模型大小...")
    original_size = calculate_model_size(model)
    print(f"   原始模型大小: {original_size:.2f} MB")
    
    print("\n3. 应用动态量化...")
    quantized_model = apply_dynamic_quantization(model, dtype=torch.qint8)
    print("✓ 量化完成")
    
    print("\n4. 计算量化模型大小...")
    quantized_size = calculate_model_size(quantized_model)
    print(f"   量化模型大小: {quantized_size:.2f} MB")
    
    print("\n5. 计算压缩比...")
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    print(f"   压缩比: {compression_ratio:.2f}x")
    print(f"   大小减少: {((1 - quantized_size/original_size) * 100):.2f}%")
    
    print("\n" + "="*80)
    print("演示完成!")
    print("="*80)
    
    print("\n接下来可以:")
    print("  - 运行完整实验: python experiments/run_all_experiments.py")
    print("  - 查看文档: docs/使用说明.md")
    print("  - 查看报告: report/课程设计报告.md")
    print()


if __name__ == "__main__":
    demo()
