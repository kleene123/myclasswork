"""
动态量化实现

提供 PyTorch 动态量化功能
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from pathlib import Path
from typing import Optional, List, Set
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    modules_to_quantize: Optional[Set[type]] = None
) -> nn.Module:
    """
    应用动态量化
    
    动态量化在推理时动态计算量化参数，主要用于权重量化。
    适用于 LSTM、Linear 等层。
    
    Args:
        model: 原始模型
        dtype: 量化数据类型 (torch.qint8 或 torch.float16)
        modules_to_quantize: 要量化的模块类型集合
        
    Returns:
        quantized_model: 量化后的模型
    """
    logger.info("开始应用动态量化...")
    
    # 设置为评估模式
    model.eval()
    
    # 默认量化 Linear 和 LSTM 层
    if modules_to_quantize is None:
        modules_to_quantize = {nn.Linear, nn.LSTM}
    
    logger.info(f"量化模块类型: {[m.__name__ for m in modules_to_quantize]}")
    logger.info(f"量化数据类型: {dtype}")
    
    # 应用动态量化
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec=modules_to_quantize,
        dtype=dtype
    )
    
    logger.info("动态量化完成")
    
    return quantized_model


def save_quantized_model(model: nn.Module, save_path: str):
    """
    保存量化模型
    
    Args:
        model: 量化模型
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存量化模型到: {save_path}")
    torch.save(model.state_dict(), save_path)
    
    # 也保存完整模型（用于后续加载）
    full_model_path = save_path.parent / f"{save_path.stem}_full.pt"
    torch.save(model, full_model_path)
    
    logger.info(f"模型已保存: {save_path}")


def load_quantized_model(load_path: str, device: str = 'cpu') -> nn.Module:
    """
    加载量化模型
    
    Args:
        load_path: 加载路径
        device: 设备
        
    Returns:
        model: 加载的量化模型
    """
    logger.info(f"加载量化模型: {load_path}")
    
    # 尝试加载完整模型
    full_model_path = Path(load_path).parent / f"{Path(load_path).stem}_full.pt"
    
    if full_model_path.exists():
        model = torch.load(full_model_path, map_location=device)
    else:
        # 如果完整模型不存在，需要先创建模型架构再加载权重
        raise FileNotFoundError(
            f"完整模型文件不存在: {full_model_path}\n"
            "请确保使用 save_quantized_model 保存模型"
        )
    
    model.eval()
    logger.info("模型加载完成")
    
    return model


def quantize_bert_dynamic(
    bert_model,
    dtype: torch.dtype = torch.qint8
):
    """
    对 BERT 模型应用动态量化
    
    Args:
        bert_model: BERT 模型实例（来自 BERTModel 类）
        dtype: 量化数据类型
        
    Returns:
        quantized_model: 量化后的模型
    """
    logger.info("对 BERT 模型应用动态量化...")
    
    model = bert_model.get_model() if hasattr(bert_model, 'get_model') else bert_model
    
    # 应用动态量化
    quantized_model = apply_dynamic_quantization(
        model,
        dtype=dtype,
        modules_to_quantize={nn.Linear}
    )
    
    return quantized_model


def evaluate_dynamic_quant(
    original_model: nn.Module,
    quantized_model: nn.Module,
    eval_fn,
    test_data
) -> dict:
    """
    评估动态量化效果
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        eval_fn: 评估函数
        test_data: 测试数据
        
    Returns:
        results: 评估结果字典
    """
    logger.info("评估动态量化效果...")
    
    # 评估原始模型
    logger.info("评估原始模型...")
    original_results = eval_fn(original_model, test_data)
    
    # 评估量化模型
    logger.info("评估量化模型...")
    quantized_results = eval_fn(quantized_model, test_data)
    
    # 计算差异
    results = {
        'original': original_results,
        'quantized': quantized_results,
        'accuracy_drop': original_results.get('accuracy', 0) - quantized_results.get('accuracy', 0)
    }
    
    logger.info(f"准确率下降: {results['accuracy_drop']:.4f}")
    
    return results


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: nn.Module
) -> dict:
    """
    比较模型大小
    
    Args:
        original_model: 原始模型
        quantized_model: 量化模型
        
    Returns:
        comparison: 大小比较结果
    """
    from ..models.model_utils import get_model_size_mb
    
    original_size = get_model_size_mb(original_model)
    quantized_size = get_model_size_mb(quantized_model)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    
    comparison = {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'size_reduction_mb': original_size - quantized_size,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    }
    
    logger.info(f"原始模型大小: {original_size:.2f} MB")
    logger.info(f"量化模型大小: {quantized_size:.2f} MB")
    logger.info(f"压缩比: {compression_ratio:.2f}x")
    logger.info(f"大小减少: {comparison['size_reduction_percent']:.2f}%")
    
    return comparison
