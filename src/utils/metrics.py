"""
评估指标计算

包括准确率、F1分数、模型大小、推理速度等
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        
    Returns:
        指标字典
    """
    # 如果是logits，取argmax
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    
    # 对于二分类和多分类，使用不同的平均方式
    num_classes = len(np.unique(labels))
    average = 'binary' if num_classes == 2 else 'macro'
    
    f1 = f1_score(labels, predictions, average=average, zero_division=0)
    precision = precision_score(labels, predictions, average=average, zero_division=0)
    recall = recall_score(labels, predictions, average=average, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


class ModelSizeCalculator:
    """模型大小计算器"""
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """
        获取模型大小
        
        Args:
            model: 模型
            
        Returns:
            大小统计字典
        """
        # 计算参数大小
        param_size = 0
        num_params = 0
        
        for param in model.parameters():
            num_params += param.numel()
            param_size += param.numel() * param.element_size()
        
        # 计算缓冲区大小
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        # 总大小（字节）
        total_size_bytes = param_size + buffer_size
        
        # 转换为不同单位
        size_mb = total_size_bytes / (1024 ** 2)
        size_kb = total_size_bytes / 1024
        
        return {
            'num_params': num_params,
            'size_bytes': total_size_bytes,
            'size_kb': size_kb,
            'size_mb': size_mb,
            'param_size_bytes': param_size,
            'buffer_size_bytes': buffer_size
        }
    
    @staticmethod
    def get_file_size(file_path: str) -> Dict[str, float]:
        """
        获取保存的模型文件大小
        
        Args:
            file_path: 文件路径
            
        Returns:
            大小统计字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 ** 2)
        size_kb = size_bytes / 1024
        
        return {
            'size_bytes': size_bytes,
            'size_kb': size_kb,
            'size_mb': size_mb
        }
    
    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, any]:
        """
        比较两个模型的大小
        
        Args:
            model1: 模型1
            model2: 模型2
            
        Returns:
            比较统计
        """
        size1 = ModelSizeCalculator.get_model_size(model1)
        size2 = ModelSizeCalculator.get_model_size(model2)
        
        compression_ratio = size1['size_mb'] / size2['size_mb'] if size2['size_mb'] > 0 else 0
        size_reduction = size1['size_mb'] - size2['size_mb']
        size_reduction_percent = (size_reduction / size1['size_mb']) * 100 if size1['size_mb'] > 0 else 0
        
        return {
            'model1_size_mb': size1['size_mb'],
            'model2_size_mb': size2['size_mb'],
            'compression_ratio': compression_ratio,
            'size_reduction_mb': size_reduction,
            'size_reduction_percent': size_reduction_percent,
            'model1_params': size1['num_params'],
            'model2_params': size2['num_params']
        }


class InferenceTimer:
    """推理速度测试器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        初始化推理计时器
        
        Args:
            model: 模型
            device: 设备
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def warmup(self, input_shape: Tuple, num_iterations: int = 10):
        """
        预热模型
        
        Args:
            input_shape: 输入形状
            num_iterations: 预热迭代次数
        """
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randint(0, 1000, input_shape).to(self.device)
                _ = self.model(dummy_input)
        
        # 同步CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    def measure_inference_time(
        self,
        dataloader,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """
        测量推理时间
        
        Args:
            dataloader: 数据加载器
            num_batches: 测试批次数
            
        Returns:
            时间统计
        """
        self.model.eval()
        
        # 预热
        first_batch = next(iter(dataloader))
        input_shape = first_batch['input_ids'].shape
        self.warmup(input_shape)
        
        # 测量
        total_time = 0.0
        total_samples = 0
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 开始计时
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # 推理
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 结束计时
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                batch_time = end_time - start_time
                total_time += batch_time
                batch_times.append(batch_time)
                total_samples += input_ids.size(0)
        
        # 计算统计
        avg_batch_time = total_time / len(batch_times) if batch_times else 0
        avg_sample_time = total_time / total_samples if total_samples > 0 else 0
        throughput = total_samples / total_time if total_time > 0 else 0
        
        batch_times = np.array(batch_times)
        
        return {
            'total_time': total_time,
            'total_samples': total_samples,
            'avg_batch_time_ms': avg_batch_time * 1000,
            'avg_sample_time_ms': avg_sample_time * 1000,
            'throughput_samples_per_sec': throughput,
            'std_batch_time_ms': batch_times.std() * 1000 if len(batch_times) > 0 else 0,
            'min_batch_time_ms': batch_times.min() * 1000 if len(batch_times) > 0 else 0,
            'max_batch_time_ms': batch_times.max() * 1000 if len(batch_times) > 0 else 0
        }
    
    def compare_inference_speed(
        self,
        other_model: nn.Module,
        dataloader,
        num_batches: int = 100
    ) -> Dict[str, any]:
        """
        比较两个模型的推理速度
        
        Args:
            other_model: 另一个模型
            dataloader: 数据加载器
            num_batches: 测试批次数
            
        Returns:
            比较统计
        """
        # 测量当前模型
        stats1 = self.measure_inference_time(dataloader, num_batches)
        
        # 测量另一个模型
        other_timer = InferenceTimer(other_model, self.device)
        stats2 = other_timer.measure_inference_time(dataloader, num_batches)
        
        # 计算加速比
        speedup = stats1['avg_batch_time_ms'] / stats2['avg_batch_time_ms'] if stats2['avg_batch_time_ms'] > 0 else 0
        
        return {
            'model1_avg_time_ms': stats1['avg_batch_time_ms'],
            'model2_avg_time_ms': stats2['avg_batch_time_ms'],
            'speedup': speedup,
            'model1_throughput': stats1['throughput_samples_per_sec'],
            'model2_throughput': stats2['throughput_samples_per_sec']
        }


def calculate_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    计算模型的稀疏度
    
    Args:
        model: 模型
        
    Returns:
        稀疏度统计
    """
    total_params = 0
    zero_params = 0
    
    layer_sparsity = {}
    
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # 只统计权重矩阵
            num_params = param.numel()
            num_zeros = (param.data == 0).sum().item()
            
            total_params += num_params
            zero_params += num_zeros
            
            layer_sparsity[name] = num_zeros / num_params if num_params > 0 else 0
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0
    
    return {
        'overall_sparsity': overall_sparsity,
        'total_params': total_params,
        'zero_params': zero_params,
        'layer_sparsity': layer_sparsity
    }
