"""
静态量化实现

提供 PyTorch 静态量化功能
"""

import torch
import torch.nn as nn
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    fuse_modules
)
import copy
from pathlib import Path
from typing import Optional, List
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def prepare_calibration_data(dataloader, num_batches: int = 100):
    """
    准备校准数据
    
    Args:
        dataloader: 数据加载器
        num_batches: 校准批次数量
        
    Returns:
        calibration_data: 校准数据列表
    """
    logger.info(f"准备校准数据，批次数: {num_batches}")
    
    calibration_data = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        calibration_data.append(batch)
    
    logger.info(f"校准数据准备完成，共 {len(calibration_data)} 个批次")
    
    return calibration_data


def apply_static_quantization(
    model: nn.Module,
    calibration_loader,
    backend: str = 'fbgemm',
    num_calibration_batches: int = 100
) -> nn.Module:
    """
    应用静态量化
    
    静态量化需要在量化前使用校准数据集确定激活值的量化参数。
    
    Args:
        model: 原始模型
        calibration_loader: 校准数据加载器
        backend: 量化后端 ('fbgemm' for x86, 'qnnpack' for ARM)
        num_calibration_batches: 校准批次数量
        
    Returns:
        quantized_model: 量化后的模型
    """
    logger.info("开始应用静态量化...")
    logger.info(f"量化后端: {backend}")
    
    # 设置量化后端
    torch.backends.quantized.engine = backend
    
    # 复制模型以避免修改原始模型
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    
    # 融合模块（可选，但推荐）
    # 注意：对于 Transformer 模型，模块融合可能有限
    logger.info("尝试融合模块...")
    try:
        model_to_quantize = fuse_modules_for_transformer(model_to_quantize)
    except Exception as e:
        logger.warning(f"模块融合失败（可以忽略）: {e}")
    
    # 设置量化配置
    model_to_quantize.qconfig = get_default_qconfig(backend)
    logger.info(f"量化配置: {model_to_quantize.qconfig}")
    
    # 准备量化模型（插入观察器）
    logger.info("准备量化模型（插入观察器）...")
    prepared_model = prepare(model_to_quantize, inplace=False)
    
    # 校准：通过校准数据集收集激活值统计信息
    logger.info("开始校准过程...")
    prepared_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            
            # 前向传播以收集统计信息
            if isinstance(batch, dict):
                # 处理字典格式的批次
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask', None)
                
                if attention_mask is not None:
                    _ = prepared_model(input_ids, attention_mask=attention_mask)
                else:
                    _ = prepared_model(input_ids)
            else:
                _ = prepared_model(batch)
            
            if (i + 1) % 10 == 0:
                logger.info(f"校准进度: {i + 1}/{num_calibration_batches}")
    
    logger.info("校准完成")
    
    # 转换为量化模型
    logger.info("转换为量化模型...")
    quantized_model = convert(prepared_model, inplace=False)
    
    logger.info("静态量化完成")
    
    return quantized_model


def fuse_modules_for_transformer(model: nn.Module) -> nn.Module:
    """
    为 Transformer 模型融合模块
    
    Args:
        model: 模型
        
    Returns:
        model: 融合后的模型
    """
    # Transformer 模型的模块融合相对有限
    # 这里提供一个基本框架，实际使用可能需要根据具体模型调整
    
    # 对于 BERT 等模型，通常不进行模块融合，因为结构较复杂
    # 如果有简单的 Linear + ReLU 组合，可以融合
    
    logger.info("Transformer 模型通常不需要模块融合")
    
    return model


def save_static_quantized_model(model: nn.Module, save_path: str):
    """
    保存静态量化模型
    
    Args:
        model: 量化模型
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存静态量化模型到: {save_path}")
    
    # 保存完整模型
    torch.save(model, save_path)
    
    # 也保存状态字典
    state_dict_path = save_path.parent / f"{save_path.stem}_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    
    logger.info(f"模型已保存: {save_path}")


def load_static_quantized_model(load_path: str, device: str = 'cpu') -> nn.Module:
    """
    加载静态量化模型
    
    Args:
        load_path: 加载路径
        device: 设备
        
    Returns:
        model: 加载的量化模型
    """
    logger.info(f"加载静态量化模型: {load_path}")
    
    model = torch.load(load_path, map_location=device)
    model.eval()
    
    logger.info("模型加载完成")
    
    return model


def quantize_bert_static(
    bert_model,
    calibration_loader,
    backend: str = 'fbgemm',
    num_calibration_batches: int = 100
):
    """
    对 BERT 模型应用静态量化
    
    Args:
        bert_model: BERT 模型实例
        calibration_loader: 校准数据加载器
        backend: 量化后端
        num_calibration_batches: 校准批次数量
        
    Returns:
        quantized_model: 量化后的模型
    """
    logger.info("对 BERT 模型应用静态量化...")
    
    model = bert_model.get_model() if hasattr(bert_model, 'get_model') else bert_model
    
    # 应用静态量化
    quantized_model = apply_static_quantization(
        model,
        calibration_loader,
        backend=backend,
        num_calibration_batches=num_calibration_batches
    )
    
    return quantized_model
