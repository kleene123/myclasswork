"""
模型工具函数

提供模型加载、保存等通用功能
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoConfig
import os


def get_model_size_mb(model: nn.Module) -> float:
    """
    计算模型大小（MB）
    
    Args:
        model: PyTorch 模型
        
    Returns:
        size_mb: 模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def count_parameters(model: nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        num_params: 参数数量
    """
    return sum(p.numel() for p in model.parameters())


def save_model(
    model: nn.Module,
    save_path: str,
    save_config: bool = True,
    config: Optional[Dict[str, Any]] = None
):
    """
    保存模型
    
    Args:
        model: 模型
        save_path: 保存路径
        save_config: 是否保存配置
        config: 配置字典
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), save_path)
    
    # 保存配置
    if save_config and config:
        config_path = save_path.parent / f"{save_path.stem}_config.pt"
        torch.save(config, config_path)


def load_model(
    model: nn.Module,
    load_path: str,
    device: str = 'cpu'
) -> nn.Module:
    """
    加载模型
    
    Args:
        model: 模型实例
        load_path: 加载路径
        device: 设备
        
    Returns:
        model: 加载后的模型
    """
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_model_file_size(model_path: str) -> float:
    """
    获取模型文件大小（MB）
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        size_mb: 文件大小（MB）
    """
    if not os.path.exists(model_path):
        return 0.0
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """
    打印模型信息
    
    Args:
        model: 模型
        model_name: 模型名称
    """
    num_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name} 信息:")
    print(f"{'='*50}")
    print(f"参数数量: {num_params:,}")
    print(f"模型大小: {size_mb:.2f} MB")
    print(f"{'='*50}\n")


def prepare_model_for_quantization(model: nn.Module) -> nn.Module:
    """
    准备模型用于量化
    
    Args:
        model: 原始模型
        
    Returns:
        model: 准备好的模型
    """
    model.eval()
    return model


def get_module_by_name(model: nn.Module, module_name: str) -> Optional[nn.Module]:
    """
    根据名称获取模块
    
    Args:
        model: 模型
        module_name: 模块名称
        
    Returns:
        module: 模块对象，如果不存在则返回 None
    """
    try:
        module = model
        for attr in module_name.split('.'):
            module = getattr(module, attr)
        return module
    except AttributeError:
        return None
