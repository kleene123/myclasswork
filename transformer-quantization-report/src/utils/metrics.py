"""
评估指标计算工具

提供各种评估指标的计算功能
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        
    Returns:
        accuracy: 准确率
    """
    return accuracy_score(labels, predictions)


def calculate_f1_score(
    predictions: List[int],
    labels: List[int],
    average: str = 'binary'
) -> float:
    """
    计算 F1 分数
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        average: 平均方式 ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        f1: F1 分数
    """
    _, _, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return f1


def calculate_precision_recall(
    predictions: List[int],
    labels: List[int],
    average: str = 'binary'
) -> tuple:
    """
    计算精确率和召回率
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        average: 平均方式
        
    Returns:
        precision: 精确率
        recall: 召回率
    """
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return precision, recall


def get_confusion_matrix(predictions: List[int], labels: List[int]) -> np.ndarray:
    """
    获取混淆矩阵
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        
    Returns:
        cm: 混淆矩阵
    """
    return confusion_matrix(labels, predictions)


def get_classification_report(
    predictions: List[int],
    labels: List[int],
    target_names: List[str] = None
) -> str:
    """
    获取分类报告
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        target_names: 类别名称列表
        
    Returns:
        report: 分类报告字符串
    """
    return classification_report(labels, predictions, target_names=target_names)


def calculate_all_metrics(
    predictions: List[int],
    labels: List[int]
) -> Dict[str, float]:
    """
    计算所有常用指标
    
    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        
    Returns:
        metrics: 包含所有指标的字典
    """
    accuracy = calculate_accuracy(predictions, labels)
    f1 = calculate_f1_score(predictions, labels)
    precision, recall = calculate_precision_recall(predictions, labels)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
        
    Returns:
        formatted: 格式化后的字符串
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)
