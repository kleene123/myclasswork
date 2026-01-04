"""
可视化工具

绘制训练曲线、性能对比图等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史字典，包含 'loss', 'accuracy' 等
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    if 'loss' in history:
        axes[0].plot(history['loss'], label='训练损失', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='验证损失', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='训练准确率', marker='o')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='验证准确率', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('训练和验证准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    
    plt.show()


def plot_comparison(
    results: List[Dict[str, any]],
    metrics: List[str] = ['accuracy', 'f1', 'model_size_mb', 'inference_time_ms'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    绘制多个模型的性能对比图
    
    Args:
        results: 结果列表，每个元素是一个包含模型名称和指标的字典
        metrics: 要对比的指标列表
        save_path: 保存路径
        figsize: 图形大小
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # 提取模型名称
    model_names = [r.get('name', f"Model {i}") for i, r in enumerate(results)]
    
    # 为每个指标绘制柱状图
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # 提取指标值
        values = []
        for r in results:
            if metric in r:
                values.append(r[metric])
            else:
                values.append(0)
        
        # 绘制柱状图
        bars = ax.bar(model_names, values, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} 对比')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()


def plot_sparsity_vs_accuracy(
    sparsity_levels: List[float],
    accuracies: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制稀疏度与准确率的关系图
    
    Args:
        sparsity_levels: 稀疏度列表
        accuracies: 准确率列表
        save_path: 保存路径
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    plt.plot(sparsity_levels, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('稀疏度')
    plt.ylabel('准确率')
    plt.title('模型稀疏度与准确率的关系')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for x, y in zip(sparsity_levels, accuracies):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"稀疏度-准确率图已保存到: {save_path}")
    
    plt.show()


def plot_model_size_vs_accuracy(
    model_sizes: List[float],
    accuracies: List[float],
    model_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    绘制模型大小与准确率的散点图
    
    Args:
        model_sizes: 模型大小列表（MB）
        accuracies: 准确率列表
        model_names: 模型名称列表
        save_path: 保存路径
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    plt.scatter(model_sizes, accuracies, s=100, alpha=0.6)
    
    # 添加模型名称标签
    for i, name in enumerate(model_names):
        plt.annotate(name, (model_sizes[i], accuracies[i]), 
                    textcoords="offset points", xytext=(5, 5), 
                    ha='left', fontsize=9)
    
    plt.xlabel('模型大小 (MB)')
    plt.ylabel('准确率')
    plt.title('模型大小与准确率的关系')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型大小-准确率图已保存到: {save_path}")
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    可视化注意力权重
    
    Args:
        attention_weights: 注意力权重矩阵 (seq_len, seq_len)
        tokens: Token列表
        layer_idx: 层索引
        head_idx: 头索引
        save_path: 保存路径
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
        square=True
    )
    
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title(f'注意力权重热力图 (Layer {layer_idx}, Head {head_idx})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力权重图已保存到: {save_path}")
    
    plt.show()


def plot_performance_heatmap(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制性能对比热力图
    
    Args:
        data: 包含性能数据的DataFrame
        save_path: 保存路径
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    # 归一化数据用于热力图
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    sns.heatmap(
        data_normalized,
        annot=data,  # 显示原始数值
        fmt='.3f',
        cmap='RdYlGn',
        cbar=True,
        linewidths=0.5
    )
    
    plt.title('模型性能对比热力图')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能热力图已保存到: {save_path}")
    
    plt.show()


def plot_pruning_progress(
    iterations: List[int],
    sparsities: List[float],
    accuracies: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    绘制渐进式剪枝进度
    
    Args:
        iterations: 迭代次数列表
        sparsities: 稀疏度列表
        accuracies: 准确率列表
        save_path: 保存路径
        figsize: 图形大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 稀疏度变化
    axes[0].plot(iterations, sparsities, marker='o', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('稀疏度')
    axes[0].set_title('剪枝进度 - 稀疏度')
    axes[0].grid(True, alpha=0.3)
    
    # 准确率变化
    axes[1].plot(iterations, accuracies, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('剪枝进度 - 准确率')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"剪枝进度图已保存到: {save_path}")
    
    plt.show()


def create_results_table(
    results: List[Dict[str, any]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    创建结果对比表格
    
    Args:
        results: 结果列表
        save_path: 保存路径（CSV）
        
    Returns:
        结果DataFrame
    """
    df = pd.DataFrame(results)
    
    if save_path:
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"结果表格已保存到: {save_path}")
    
    return df
