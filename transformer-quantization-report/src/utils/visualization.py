"""
可视化工具

提供各种图表生成功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_accuracy_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "准确率对比"
):
    """
    绘制准确率对比柱状图
    
    Args:
        results: 模型名称到准确率的字典
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    models = list(results.keys())
    accuracies = [results[m] * 100 for m in models]
    
    colors = sns.color_palette("husl", len(models))
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_size_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "模型大小对比"
):
    """
    绘制模型大小对比柱状图
    
    Args:
        results: 模型名称到大小(MB)的字典
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    models = list(results.keys())
    sizes = [results[m] for m in models]
    
    colors = sns.color_palette("rocket", len(models))
    bars = plt.bar(models, sizes, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{size:.1f} MB', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('模型大小 (MB)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_speed_comparison(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "推理速度对比"
):
    """
    绘制推理速度对比柱状图
    
    Args:
        results: 模型名称到推理时间(ms)的字典
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    models = list(results.keys())
    times = [results[m] for m in models]
    
    colors = sns.color_palette("mako", len(models))
    bars = plt.bar(models, times, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time:.1f} ms', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('推理时间 (ms)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_radar_chart(
    data: Dict[str, Dict[str, float]],
    categories: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    title: str = "综合性能雷达图"
):
    """
    绘制综合性能雷达图
    
    Args:
        data: 模型名称到指标字典的映射
        categories: 评估指标类别列表
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("Set2", len(data))
    
    for (model_name, metrics), color in zip(data.items(), colors):
        values = [metrics.get(cat, 0) for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_compression_ratio(
    results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "压缩比对比"
):
    """
    绘制压缩比对比图
    
    Args:
        results: 模型名称到压缩比的字典
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    models = list(results.keys())
    ratios = [results[m] for m in models]
    
    colors = sns.color_palette("coolwarm", len(models))
    bars = plt.barh(models, ratios, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, ratio in zip(bars, ratios):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{ratio:.2f}x', ha='left', va='center', fontsize=10)
    
    plt.xlabel('压缩比', fontsize=12)
    plt.ylabel('模型', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_accuracy_vs_size(
    data: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    title: str = "准确率-模型大小权衡"
):
    """
    绘制准确率vs模型大小散点图
    
    Args:
        data: 模型数据 {model_name: {'accuracy': x, 'size': y}}
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    models = list(data.keys())
    accuracies = [data[m]['accuracy'] * 100 for m in models]
    sizes = [data[m]['size'] for m in models]
    
    colors = sns.color_palette("viridis", len(models))
    
    plt.scatter(sizes, accuracies, c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for model, size, acc in zip(models, sizes, accuracies):
        plt.annotate(model, (size, acc), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('模型大小 (MB)', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    title: str = "训练曲线"
):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    # 如果提供了准确率，绘制双子图
    has_accuracies = (train_accuracies and len(train_accuracies) > 0) or (val_accuracies and len(val_accuracies) > 0)
    if has_accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        ax1.plot(epochs, train_losses, 'o-', label='训练损失', linewidth=2, markersize=6)
        if val_losses:
            ax1.plot(epochs, val_losses, 's-', label='验证损失', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.set_title('损失曲线', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # 准确率曲线
        if train_accuracies:
            ax2.plot(epochs, train_accuracies, 'o-', label='训练准确率', linewidth=2, markersize=6)
        if val_accuracies:
            ax2.plot(epochs, val_accuracies, 's-', label='验证准确率', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_title('准确率曲线', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
    else:
        # 只绘制损失曲线
        plt.figure(figsize=figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'o-', label='训练损失', linewidth=2, markersize=6)
        if val_losses:
            plt.plot(epochs, val_losses, 's-', label='验证损失', linewidth=2, markersize=6)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    title: str = "混淆矩阵"
):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图片大小
        title: 图表标题
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数'})
    
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def generate_summary_dashboard(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12)
):
    """
    生成综合仪表板
    
    Args:
        results: 完整的实验结果数据
        save_path: 保存路径
        figsize: 图片大小
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 提取数据
    models = list(results.keys())
    accuracies = {m: results[m].get('accuracy', 0) for m in models}
    sizes = {m: results[m].get('size', 0) for m in models}
    speeds = {m: results[m].get('inference_time', 0) for m in models}
    
    # 准确率对比
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(models, [accuracies[m] * 100 for m in models], 
                   color=sns.color_palette("husl", len(models)), alpha=0.8)
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('准确率对比', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 模型大小对比
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(models, [sizes[m] for m in models],
                   color=sns.color_palette("rocket", len(models)), alpha=0.8)
    ax2.set_ylabel('模型大小 (MB)')
    ax2.set_title('模型大小对比', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 推理速度对比
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(models, [speeds[m] for m in models],
                   color=sns.color_palette("mako", len(models)), alpha=0.8)
    ax3.set_ylabel('推理时间 (ms)')
    ax3.set_title('推理速度对比', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 准确率 vs 大小散点图
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter([sizes[m] for m in models], [accuracies[m] * 100 for m in models],
               c=sns.color_palette("viridis", len(models)), s=200, alpha=0.6, 
               edgecolors='black', linewidth=1.5)
    for model in models:
        ax4.annotate(model, (sizes[model], accuracies[model] * 100),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('模型大小 (MB)')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_title('准确率-大小权衡', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 综合数据表
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for model in models:
        row = [
            model,
            f"{accuracies[model]*100:.2f}%",
            f"{sizes[model]:.1f} MB",
            f"{speeds[model]:.1f} ms"
        ]
        table_data.append(row)
    
    table = ax5.table(cellText=table_data,
                     colLabels=['模型', '准确率', '模型大小', '推理时间'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('量化实验综合仪表板', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
