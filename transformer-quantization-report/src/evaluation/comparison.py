"""
对比分析工具

提供多模型综合对比分析功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from ..utils.logger import setup_logger
from ..utils.visualization import (
    plot_accuracy_comparison,
    plot_size_comparison,
    plot_speed_comparison,
    plot_radar_chart,
    plot_compression_ratio,
    plot_accuracy_vs_size,
    generate_summary_dashboard
)
from .accuracy_eval import evaluate_accuracy
from .performance_eval import measure_inference_time, measure_memory_usage
from .size_eval import calculate_model_size, calculate_compression_ratio

logger = setup_logger(__name__)


def compare_models(
    models: Dict[str, nn.Module],
    dataloader: DataLoader,
    device: str = 'cpu',
    num_iterations: int = 100,
    baseline_name: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    全面比较多个模型
    
    Args:
        models: 模型字典 {name: model}
        dataloader: 测试数据加载器
        device: 设备
        num_iterations: 性能测试迭代次数
        baseline_name: 基线模型名称
        
    Returns:
        results: 完整的比较结果
    """
    logger.info(f"开始全面比较 {len(models)} 个模型...")
    logger.info(f"模型列表: {list(models.keys())}")
    
    results = {}
    baseline_size = None
    
    # 获取基线大小
    if baseline_name and baseline_name in models:
        baseline_size = calculate_model_size(models[baseline_name])
    
    for name, model in models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"评估模型: {name}")
        logger.info(f"{'='*60}")
        
        # 准确率评估
        logger.info("评估准确率...")
        accuracy_metrics = evaluate_accuracy(model, dataloader, device, desc=f"{name} - 准确率")
        
        # 模型大小
        logger.info("计算模型大小...")
        size_mb = calculate_model_size(model)
        
        # 性能评估
        logger.info("评估推理性能...")
        timing_metrics = measure_inference_time(
            model, dataloader, device, 
            num_warmup=10, num_iterations=num_iterations
        )
        
        logger.info("评估内存占用...")
        memory_metrics = measure_memory_usage(model, dataloader, device)
        
        # 整合结果
        model_results = {
            'accuracy': accuracy_metrics['accuracy'],
            'f1_score': accuracy_metrics['f1_score'],
            'precision': accuracy_metrics['precision'],
            'recall': accuracy_metrics['recall'],
            'size_mb': size_mb,
            'inference_time_ms': timing_metrics['mean_ms'],
            'inference_time_std_ms': timing_metrics['std_ms'],
            'memory_used_mb': memory_metrics['memory_used_mb']
        }
        
        # 计算压缩比和加速比
        if baseline_size and baseline_size > 0:
            compression_ratio = baseline_size / size_mb if size_mb > 0 else 0
            model_results['compression_ratio'] = compression_ratio
        
        results[name] = model_results
        
        logger.info(f"\n{name} 结果:")
        logger.info(f"  准确率: {model_results['accuracy']:.4f}")
        logger.info(f"  F1 分数: {model_results['f1_score']:.4f}")
        logger.info(f"  模型大小: {model_results['size_mb']:.2f} MB")
        logger.info(f"  推理时间: {model_results['inference_time_ms']:.2f} ms")
        logger.info(f"  内存占用: {model_results['memory_used_mb']:.2f} MB")
        if 'compression_ratio' in model_results:
            logger.info(f"  压缩比: {model_results['compression_ratio']:.2f}x")
    
    # 计算相对指标
    if baseline_name and baseline_name in results:
        baseline = results[baseline_name]
        
        for name in results:
            if name != baseline_name:
                # 准确率下降
                results[name]['accuracy_drop'] = baseline['accuracy'] - results[name]['accuracy']
                results[name]['accuracy_drop_percent'] = (
                    results[name]['accuracy_drop'] / baseline['accuracy'] * 100
                ) if baseline['accuracy'] > 0 else 0
                
                # 加速比
                results[name]['speedup'] = (
                    baseline['inference_time_ms'] / results[name]['inference_time_ms']
                ) if results[name]['inference_time_ms'] > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info("全面比较完成")
    logger.info(f"{'='*60}\n")
    
    return results


def generate_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    生成对比表格
    
    Args:
        results: 比较结果
        save_path: 保存路径（可选）
        
    Returns:
        df: 对比表格 DataFrame
    """
    logger.info("生成对比表格...")
    
    # 准备数据
    table_data = []
    
    for model_name, metrics in results.items():
        row = {
            '模型': model_name,
            '准确率': f"{metrics['accuracy']*100:.2f}%",
            'F1分数': f"{metrics['f1_score']:.4f}",
            '模型大小(MB)': f"{metrics['size_mb']:.1f}",
            '推理时间(ms)': f"{metrics['inference_time_ms']:.1f}",
            '内存占用(MB)': f"{metrics['memory_used_mb']:.1f}"
        }
        
        if 'compression_ratio' in metrics:
            row['压缩比'] = f"{metrics['compression_ratio']:.2f}x"
        
        if 'speedup' in metrics:
            row['加速比'] = f"{metrics['speedup']:.2f}x"
        
        if 'accuracy_drop' in metrics:
            row['准确率下降'] = f"{metrics['accuracy_drop']*100:.2f}%"
        
        table_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(table_data)
    
    # 保存
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"对比表格已保存到: {save_path}")
    
    return df


def generate_comparison_plots(
    results: Dict[str, Dict[str, float]],
    save_dir: str
):
    """
    生成对比图表
    
    Args:
        results: 比较结果
        save_dir: 保存目录
    """
    logger.info("生成对比图表...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 准确率对比图
    accuracies = {name: metrics['accuracy'] for name, metrics in results.items()}
    plot_accuracy_comparison(
        accuracies,
        save_path=str(save_dir / 'accuracy_comparison.png'),
        title='准确率对比'
    )
    logger.info("✓ 准确率对比图已生成")
    
    # 模型大小对比图
    sizes = {name: metrics['size_mb'] for name, metrics in results.items()}
    plot_size_comparison(
        sizes,
        save_path=str(save_dir / 'size_comparison.png'),
        title='模型大小对比'
    )
    logger.info("✓ 模型大小对比图已生成")
    
    # 推理速度对比图
    speeds = {name: metrics['inference_time_ms'] for name, metrics in results.items()}
    plot_speed_comparison(
        speeds,
        save_path=str(save_dir / 'speed_comparison.png'),
        title='推理速度对比'
    )
    logger.info("✓ 推理速度对比图已生成")
    
    # 压缩比对比图
    if any('compression_ratio' in metrics for metrics in results.values()):
        compression_ratios = {
            name: metrics.get('compression_ratio', 1.0) 
            for name, metrics in results.items()
        }
        plot_compression_ratio(
            compression_ratios,
            save_path=str(save_dir / 'compression_ratio.png'),
            title='压缩比对比'
        )
        logger.info("✓ 压缩比对比图已生成")
    
    # 准确率-大小权衡图
    tradeoff_data = {
        name: {
            'accuracy': metrics['accuracy'],
            'size': metrics['size_mb']
        }
        for name, metrics in results.items()
    }
    plot_accuracy_vs_size(
        tradeoff_data,
        save_path=str(save_dir / 'accuracy_vs_size.png'),
        title='准确率-模型大小权衡'
    )
    logger.info("✓ 准确率-大小权衡图已生成")
    
    # 综合性能雷达图
    # 标准化指标用于雷达图
    radar_data = {}
    
    # 找到最大值用于归一化
    max_acc = max(m['accuracy'] for m in results.values())
    max_size = max(m['size_mb'] for m in results.values())
    max_time = max(m['inference_time_ms'] for m in results.values())
    
    for name, metrics in results.items():
        radar_data[name] = {
            '准确率': metrics['accuracy'] / max_acc if max_acc > 0 else 0,
            '模型压缩': (1 - metrics['size_mb'] / max_size) if max_size > 0 else 0,  # 越小越好
            '推理速度': (1 - metrics['inference_time_ms'] / max_time) if max_time > 0 else 0,  # 越小越好
            'F1分数': metrics['f1_score'] / max(m['f1_score'] for m in results.values()) if max(m['f1_score'] for m in results.values()) > 0 else 0
        }
    
    plot_radar_chart(
        radar_data,
        categories=['准确率', '模型压缩', '推理速度', 'F1分数'],
        save_path=str(save_dir / 'radar_chart.png'),
        title='综合性能雷达图'
    )
    logger.info("✓ 综合性能雷达图已生成")
    
    # 综合仪表板
    generate_summary_dashboard(
        results,
        save_path=str(save_dir / 'summary_dashboard.png')
    )
    logger.info("✓ 综合仪表板已生成")
    
    logger.info(f"所有对比图表已保存到: {save_dir}")


def export_results(
    results: Dict[str, Dict[str, float]],
    save_dir: str,
    format: str = 'csv'
):
    """
    导出实验结果
    
    Args:
        results: 实验结果
        save_dir: 保存目录
        format: 导出格式 ('csv', 'json', 'excel')
    """
    logger.info(f"导出实验结果（格式: {format}）...")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        # CSV 格式
        df = generate_comparison_table(results)
        save_path = save_dir / 'results_summary.csv'
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"结果已导出到: {save_path}")
    
    elif format == 'json':
        # JSON 格式
        import json
        save_path = save_dir / 'results_summary.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已导出到: {save_path}")
    
    elif format == 'excel':
        # Excel 格式
        df = generate_comparison_table(results)
        save_path = save_dir / 'results_summary.xlsx'
        df.to_excel(save_path, index=False, engine='openpyxl')
        logger.info(f"结果已导出到: {save_path}")
    
    else:
        logger.warning(f"不支持的格式: {format}")


def generate_summary_report(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    baseline_name: Optional[str] = None
):
    """
    生成文字总结报告
    
    Args:
        results: 实验结果
        save_path: 保存路径
        baseline_name: 基线模型名称
    """
    logger.info("生成总结报告...")
    
    report_lines = []
    report_lines.append("# 量化实验结果总结报告\n")
    report_lines.append(f"## 实验模型数量: {len(results)}\n")
    
    # 基线模型信息
    if baseline_name and baseline_name in results:
        baseline = results[baseline_name]
        report_lines.append(f"## 基线模型: {baseline_name}\n")
        report_lines.append(f"- 准确率: {baseline['accuracy']*100:.2f}%")
        report_lines.append(f"- 模型大小: {baseline['size_mb']:.2f} MB")
        report_lines.append(f"- 推理时间: {baseline['inference_time_ms']:.2f} ms\n")
    
    # 各模型详细信息
    report_lines.append("## 各模型详细结果\n")
    
    for name, metrics in results.items():
        report_lines.append(f"### {name}\n")
        report_lines.append(f"- 准确率: {metrics['accuracy']*100:.2f}%")
        report_lines.append(f"- F1 分数: {metrics['f1_score']:.4f}")
        report_lines.append(f"- 模型大小: {metrics['size_mb']:.2f} MB")
        report_lines.append(f"- 推理时间: {metrics['inference_time_ms']:.2f} ms")
        report_lines.append(f"- 内存占用: {metrics['memory_used_mb']:.2f} MB")
        
        if 'compression_ratio' in metrics:
            report_lines.append(f"- 压缩比: {metrics['compression_ratio']:.2f}x")
        if 'speedup' in metrics:
            report_lines.append(f"- 加速比: {metrics['speedup']:.2f}x")
        if 'accuracy_drop' in metrics:
            report_lines.append(f"- 准确率下降: {metrics['accuracy_drop']*100:.2f}%")
        
        report_lines.append("")
    
    # 保存报告
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"总结报告已保存到: {save_path}")
