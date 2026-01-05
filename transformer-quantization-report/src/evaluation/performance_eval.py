"""
性能评估模块

提供推理速度、内存占用等性能指标评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import psutil
import os
from typing import Dict, Optional, List
from tqdm import tqdm
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def measure_inference_time(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    num_warmup: int = 10,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    测量推理时间
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_warmup: 预热次数
        num_iterations: 测试迭代次数
        
    Returns:
        timing_results: 时间统计结果
    """
    logger.info("测量推理时间...")
    
    model.eval()
    model.to(device)
    
    # 准备测试数据
    test_batches = []
    for i, batch in enumerate(dataloader):
        test_batches.append(batch)
        if i >= num_warmup + num_iterations:
            break
    
    if len(test_batches) < num_warmup + num_iterations:
        logger.warning(f"数据不足，仅有 {len(test_batches)} 个批次")
        num_iterations = max(1, len(test_batches) - num_warmup)
    
    # 预热
    logger.info(f"预热 {num_warmup} 次...")
    with torch.no_grad():
        for i in range(min(num_warmup, len(test_batches))):
            batch = test_batches[i]
            _ = _run_inference(model, batch, device)
    
    # 测试
    logger.info(f"测试 {num_iterations} 次...")
    times = []
    
    with torch.no_grad():
        for i in range(num_warmup, min(num_warmup + num_iterations, len(test_batches))):
            batch = test_batches[i]
            
            start_time = time.time()
            _ = _run_inference(model, batch, device)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            times.append(inference_time)
    
    # 统计
    import numpy as np
    timing_results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'num_iterations': len(times)
    }
    
    logger.info(f"平均推理时间: {timing_results['mean_ms']:.2f} ms")
    logger.info(f"标准差: {timing_results['std_ms']:.2f} ms")
    
    return timing_results


def _run_inference(model: nn.Module, batch, device: str):
    """运行单次推理"""
    if isinstance(batch, dict):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask')
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids)
    else:
        inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        outputs = model(inputs)
    
    return outputs


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    duration_seconds: int = 10
) -> Dict[str, float]:
    """
    测量吞吐量（每秒处理样本数）
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        duration_seconds: 测试持续时间（秒）
        
    Returns:
        throughput_results: 吞吐量结果
    """
    logger.info(f"测量吞吐量（持续 {duration_seconds} 秒）...")
    
    model.eval()
    model.to(device)
    
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        while (time.time() - start_time) < duration_seconds:
            for batch in dataloader:
                _ = _run_inference(model, batch, device)
                
                # 计算批次大小
                if isinstance(batch, dict):
                    batch_size = batch['input_ids'].size(0)
                else:
                    batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
                
                total_samples += batch_size
                
                if (time.time() - start_time) >= duration_seconds:
                    break
    
    elapsed_time = time.time() - start_time
    throughput = total_samples / elapsed_time
    
    throughput_results = {
        'samples_per_second': throughput,
        'total_samples': total_samples,
        'elapsed_seconds': elapsed_time
    }
    
    logger.info(f"吞吐量: {throughput:.2f} 样本/秒")
    
    return throughput_results


def measure_memory_usage(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    测量内存占用
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        memory_results: 内存占用结果
    """
    logger.info("测量内存占用...")
    
    model.eval()
    model.to(device)
    
    process = psutil.Process(os.getpid())
    
    # 获取基线内存
    baseline_memory = process.memory_info().rss / (1024 ** 2)  # MB
    
    # 如果使用 GPU
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        baseline_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
    
    # 运行推理
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            _ = _run_inference(model, batch, device)
            
            if i >= 10:  # 只运行几个批次
                break
    
    # 获取峰值内存
    peak_memory = process.memory_info().rss / (1024 ** 2)  # MB
    memory_used = peak_memory - baseline_memory
    
    memory_results = {
        'baseline_memory_mb': baseline_memory,
        'peak_memory_mb': peak_memory,
        'memory_used_mb': memory_used
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        gpu_memory_used = peak_gpu_memory - baseline_gpu_memory
        
        memory_results['baseline_gpu_memory_mb'] = baseline_gpu_memory
        memory_results['peak_gpu_memory_mb'] = peak_gpu_memory
        memory_results['gpu_memory_used_mb'] = gpu_memory_used
        
        logger.info(f"GPU 内存占用: {gpu_memory_used:.2f} MB")
    
    logger.info(f"CPU 内存占用: {memory_used:.2f} MB")
    
    return memory_results


def measure_latency(
    model: nn.Module,
    sample_input,
    device: str = 'cpu',
    num_runs: int = 100
) -> Dict[str, float]:
    """
    测量单样本延迟
    
    Args:
        model: 模型
        sample_input: 样本输入
        device: 设备
        num_runs: 运行次数
        
    Returns:
        latency_results: 延迟结果
    """
    logger.info(f"测量延迟（{num_runs} 次运行）...")
    
    model.eval()
    model.to(device)
    
    latencies = []
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = _run_inference(model, sample_input, device)
    
    # 测试
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = _run_inference(model, sample_input, device)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # 毫秒
            latencies.append(latency)
    
    import numpy as np
    latency_results = {
        'mean_latency_ms': float(np.mean(latencies)),
        'std_latency_ms': float(np.std(latencies)),
        'min_latency_ms': float(np.min(latencies)),
        'max_latency_ms': float(np.max(latencies)),
        'p50_latency_ms': float(np.percentile(latencies, 50)),
        'p95_latency_ms': float(np.percentile(latencies, 95)),
        'p99_latency_ms': float(np.percentile(latencies, 99))
    }
    
    logger.info(f"平均延迟: {latency_results['mean_latency_ms']:.2f} ms")
    logger.info(f"P95 延迟: {latency_results['p95_latency_ms']:.2f} ms")
    
    return latency_results


def compare_performance(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: str = 'cpu',
    num_iterations: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的性能
    
    Args:
        models: 模型字典
        dataloader: 数据加载器
        device: 设备
        num_iterations: 测试迭代次数
        
    Returns:
        results: 性能比较结果
    """
    logger.info(f"开始比较 {len(models)} 个模型的性能...")
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n评估模型: {name}")
        
        timing = measure_inference_time(model, dataloader, device, num_iterations=num_iterations)
        memory = measure_memory_usage(model, dataloader, device)
        
        results[name] = {
            'inference_time_ms': timing['mean_ms'],
            'inference_time_std_ms': timing['std_ms'],
            'memory_used_mb': memory['memory_used_mb']
        }
        
        logger.info(f"{name} - 推理时间: {timing['mean_ms']:.2f} ms, 内存: {memory['memory_used_mb']:.2f} MB")
    
    return results
