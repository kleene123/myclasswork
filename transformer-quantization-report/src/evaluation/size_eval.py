"""
模型大小评估模块

提供模型大小、参数量、压缩比等评估功能
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from ..utils.logger import setup_logger
from ..models.model_utils import get_model_size_mb, count_parameters, get_model_file_size

logger = setup_logger(__name__)


def calculate_model_size(model: nn.Module) -> float:
    """
    计算模型大小（MB）
    
    Args:
        model: 模型
        
    Returns:
        size_mb: 模型大小（MB）
    """
    size_mb = get_model_size_mb(model)
    logger.info(f"模型大小: {size_mb:.2f} MB")
    return size_mb


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: 模型
        
    Returns:
        param_info: 参数信息字典
    """
    total_params = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    param_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params
    }
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"不可训练参数: {non_trainable_params:,}")
    
    return param_info


def calculate_compression_ratio(
    original_model: nn.Module,
    quantized_model: nn.Module
) -> float:
    """
    计算压缩比
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        
    Returns:
        compression_ratio: 压缩比
    """
    original_size = get_model_size_mb(original_model)
    quantized_size = get_model_size_mb(quantized_model)
    
    if quantized_size > 0:
        compression_ratio = original_size / quantized_size
    else:
        compression_ratio = 0.0
    
    logger.info(f"原始模型: {original_size:.2f} MB")
    logger.info(f"量化模型: {quantized_size:.2f} MB")
    logger.info(f"压缩比: {compression_ratio:.2f}x")
    
    return compression_ratio


def compare_model_sizes(
    models: Dict[str, nn.Module],
    baseline_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的大小
    
    Args:
        models: 模型字典 {name: model}
        baseline_name: 基线模型名称（用于计算压缩比）
        
    Returns:
        results: 大小比较结果
    """
    logger.info(f"比较 {len(models)} 个模型的大小...")
    
    results = {}
    baseline_size = None
    
    # 如果指定了基线，先获取基线大小
    if baseline_name and baseline_name in models:
        baseline_size = get_model_size_mb(models[baseline_name])
    
    for name, model in models.items():
        size_mb = get_model_size_mb(model)
        params = count_parameters(model)
        
        result = {
            'size_mb': size_mb,
            'parameters': params
        }
        
        # 计算相对于基线的压缩比
        if baseline_size and baseline_size > 0:
            if name == baseline_name:
                result['compression_ratio'] = 1.0
            else:
                result['compression_ratio'] = baseline_size / size_mb if size_mb > 0 else 0.0
        
        results[name] = result
        
        logger.info(f"{name}: {size_mb:.2f} MB, {params:,} 参数")
    
    return results


def get_detailed_size_info(model: nn.Module) -> Dict[str, any]:
    """
    获取详细的模型大小信息
    
    Args:
        model: 模型
        
    Returns:
        info: 详细信息字典
    """
    # 计算各部分大小
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    info = {
        'total_size_mb': total_size / (1024 ** 2),
        'parameters_size_mb': param_size / (1024 ** 2),
        'buffers_size_mb': buffer_size / (1024 ** 2),
        'total_parameters': count_parameters(model),
        'parameter_breakdown': {}
    }
    
    # 分析各层参数大小
    layer_sizes = {}
    for name, param in model.named_parameters():
        layer_name = name.split('.')[0]  # 获取顶层模块名
        size = param.numel() * param.element_size() / (1024 ** 2)  # MB
        
        if layer_name in layer_sizes:
            layer_sizes[layer_name] += size
        else:
            layer_sizes[layer_name] = size
    
    info['parameter_breakdown'] = layer_sizes
    
    logger.info(f"总大小: {info['total_size_mb']:.2f} MB")
    logger.info(f"参数大小: {info['parameters_size_mb']:.2f} MB")
    logger.info(f"缓冲区大小: {info['buffers_size_mb']:.2f} MB")
    
    return info


def save_size_comparison_csv(
    results: Dict[str, Dict[str, float]],
    save_path: str
):
    """
    保存大小比较结果为 CSV
    
    Args:
        results: 比较结果
        save_path: 保存路径
    """
    import pandas as pd
    
    # 转换为 DataFrame
    data = []
    for model_name, metrics in results.items():
        row = {'模型': model_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"大小比较结果已保存到: {save_path}")


def analyze_quantization_savings(
    original_size_mb: float,
    quantized_size_mb: float
) -> Dict[str, float]:
    """
    分析量化带来的空间节省
    
    Args:
        original_size_mb: 原始模型大小（MB）
        quantized_size_mb: 量化模型大小（MB）
        
    Returns:
        savings: 节省信息
    """
    size_reduction = original_size_mb - quantized_size_mb
    reduction_percent = (size_reduction / original_size_mb * 100) if original_size_mb > 0 else 0
    compression_ratio = (original_size_mb / quantized_size_mb) if quantized_size_mb > 0 else 0
    
    savings = {
        'original_size_mb': original_size_mb,
        'quantized_size_mb': quantized_size_mb,
        'size_reduction_mb': size_reduction,
        'reduction_percent': reduction_percent,
        'compression_ratio': compression_ratio
    }
    
    logger.info(f"原始大小: {original_size_mb:.2f} MB")
    logger.info(f"量化后大小: {quantized_size_mb:.2f} MB")
    logger.info(f"减少: {size_reduction:.2f} MB ({reduction_percent:.2f}%)")
    logger.info(f"压缩比: {compression_ratio:.2f}x")
    
    return savings
