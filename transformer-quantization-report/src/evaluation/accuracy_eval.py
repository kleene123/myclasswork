"""
准确率评估模块

提供模型准确率评估功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from tqdm import tqdm
from ..utils.logger import setup_logger
from ..utils.metrics import calculate_all_metrics

logger = setup_logger(__name__)


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    desc: str = "评估准确率"
) -> Dict[str, float]:
    """
    评估模型准确率
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        desc: 进度条描述
        
    Returns:
        metrics: 包含准确率等指标的字典
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # 准备输入
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask')
                labels = batch['labels'].to(device)
                
                # 前向传播
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            # 获取预测
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    # 计算指标
    metrics = calculate_all_metrics(all_predictions, all_labels)
    
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"F1 分数: {metrics['f1_score']:.4f}")
    
    return metrics


def evaluate_f1_score(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    评估 F1 分数
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        f1: F1 分数
    """
    metrics = evaluate_accuracy(model, dataloader, device, desc="评估 F1 分数")
    return metrics['f1_score']


def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    评估困惑度（用于语言模型）
    
    Args:
        model: 语言模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        perplexity: 困惑度
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估困惑度"):
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
                
                outputs = model(input_ids)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            # 获取 logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += labels.numel()
    
    # 计算困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"困惑度: {perplexity:.4f}")
    
    return perplexity


def compare_accuracies(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的准确率
    
    Args:
        models: 模型字典 {name: model}
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        results: 每个模型的评估结果
    """
    logger.info(f"开始比较 {len(models)} 个模型的准确率...")
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\n评估模型: {name}")
        metrics = evaluate_accuracy(model, dataloader, device, desc=f"评估 {name}")
        results[name] = metrics
        
        logger.info(f"{name} - 准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return results


def get_predictions_and_labels(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> tuple:
    """
    获取所有预测和标签
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        predictions: 预测列表
        labels: 标签列表
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="获取预测"):
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask')
                labels = batch['labels'].to(device)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    return all_predictions, all_labels
