"""
结果对比分析

对比所有实验的结果并生成可视化图表
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    plot_comparison,
    plot_sparsity_vs_accuracy,
    plot_model_size_vs_accuracy,
    create_results_table,
    plot_performance_heatmap
)


def load_results(results_dir: str):
    """加载所有实验结果"""
    results = {
        'baseline': None,
        'pruning': None,
        'quantization': None,
        'combined': None
    }
    
    # 加载基线结果
    baseline_path = os.path.join(results_dir, '../outputs/baseline_results.json')
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            results['baseline'] = json.load(f)
    
    # 加载剪枝结果
    pruning_path = os.path.join(results_dir, 'pruning_results.json')
    if os.path.exists(pruning_path):
        with open(pruning_path, 'r', encoding='utf-8') as f:
            results['pruning'] = json.load(f)
    
    # 加载量化结果
    quant_path = os.path.join(results_dir, 'quantization_results.json')
    if os.path.exists(quant_path):
        with open(quant_path, 'r', encoding='utf-8') as f:
            results['quantization'] = json.load(f)
    
    # 加载组合结果
    combined_path = os.path.join(results_dir, 'combined_results.json')
    if os.path.exists(combined_path):
        with open(combined_path, 'r', encoding='utf-8') as f:
            results['combined'] = json.load(f)
    
    return results


def create_comparison_data(results):
    """创建对比数据"""
    comparison_data = []
    
    # 添加基线
    if results['baseline']:
        comparison_data.append({
            'name': 'Baseline',
            'accuracy': results['baseline'].get('accuracy', 0),
            'f1': results['baseline'].get('f1', 0),
            'model_size_mb': results['baseline'].get('model_size_mb', 0)
        })
    
    # 添加剪枝结果
    if results['pruning']:
        for item in results['pruning']:
            if 'error' not in item:
                name = f"{item['method']}"
                if 'target_sparsity' in item:
                    name += f" ({item['target_sparsity']:.1%})"
                elif 'num_heads_pruned' in item:
                    name += f" ({item['num_heads_pruned']} heads)"
                
                comparison_data.append({
                    'name': name,
                    'accuracy': item.get('accuracy', 0),
                    'f1': item.get('f1', 0),
                    'model_size_mb': item.get('model_size_mb', 0)
                })
    
    # 添加量化结果
    if results['quantization']:
        for item in results['quantization']:
            if 'error' not in item:
                comparison_data.append({
                    'name': item['method'],
                    'accuracy': item.get('accuracy', 0),
                    'f1': item.get('f1', 0),
                    'model_size_mb': item.get('model_size_mb', 0)
                })
    
    # 添加组合结果
    if results['combined']:
        for item in results['combined']:
            if 'error' not in item:
                comparison_data.append({
                    'name': item['strategy'],
                    'accuracy': item.get('accuracy_final', 0),
                    'f1': item.get('f1_final', 0),
                    'model_size_mb': item.get('model_size_mb', 0)
                })
    
    return comparison_data


def main(args):
    """主函数"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    print("加载实验结果...")
    results = load_results(args.results_dir)
    
    # 创建对比数据
    comparison_data = create_comparison_data(results)
    
    if not comparison_data:
        print("没有找到实验结果！")
        return
    
    print(f"找到 {len(comparison_data)} 个实验结果")
    
    # 创建结果表格
    print("\n生成结果表格...")
    df = create_results_table(
        comparison_data,
        save_path=os.path.join(args.output_dir, 'comparison_table.csv')
    )
    
    print("\n结果表格:")
    print(df.to_string(index=False))
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(
        comparison_data,
        metrics=['accuracy', 'f1', 'model_size_mb'],
        save_path=os.path.join(args.output_dir, 'comparison_plot.png')
    )
    
    # 绘制稀疏度vs准确率（如果有剪枝结果）
    if results['pruning']:
        pruning_data = [item for item in results['pruning'] 
                       if 'error' not in item and 'actual_sparsity' in item]
        if pruning_data:
            sparsities = [item['actual_sparsity'] for item in pruning_data]
            accuracies = [item['accuracy'] for item in pruning_data]
            
            plot_sparsity_vs_accuracy(
                sparsities,
                accuracies,
                save_path=os.path.join(args.output_dir, 'sparsity_vs_accuracy.png')
            )
    
    # 绘制模型大小vs准确率
    model_sizes = [item['model_size_mb'] for item in comparison_data]
    accuracies = [item['accuracy'] for item in comparison_data]
    names = [item['name'] for item in comparison_data]
    
    plot_model_size_vs_accuracy(
        model_sizes,
        accuracies,
        names,
        save_path=os.path.join(args.output_dir, 'size_vs_accuracy.png')
    )
    
    # 生成性能热力图
    if len(comparison_data) > 1:
        df_normalized = df[['accuracy', 'f1', 'model_size_mb']].copy()
        df_normalized.index = df['name']
        
        plot_performance_heatmap(
            df_normalized,
            save_path=os.path.join(args.output_dir, 'performance_heatmap.png')
        )
    
    # 生成总结报告
    print("\n生成总结报告...")
    summary = {
        'total_experiments': len(comparison_data),
        'best_accuracy': max(comparison_data, key=lambda x: x['accuracy']),
        'smallest_model': min(comparison_data, key=lambda x: x['model_size_mb']),
        'best_tradeoff': None
    }
    
    # 计算最佳权衡（准确率和模型大小的综合）
    if results['baseline']:
        baseline_acc = results['baseline'].get('accuracy', 0)
        baseline_size = results['baseline'].get('model_size_mb', 1)
        
        scores = []
        for item in comparison_data:
            # 综合得分：保持准确率的同时减小模型大小
            acc_retention = item['accuracy'] / baseline_acc if baseline_acc > 0 else 0
            size_reduction = baseline_size / item['model_size_mb'] if item['model_size_mb'] > 0 else 0
            score = acc_retention * 0.6 + size_reduction * 0.4  # 权重可调
            scores.append((item, score))
        
        if scores:
            summary['best_tradeoff'] = max(scores, key=lambda x: x[1])[0]
    
    # 保存总结
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_experiments': summary['total_experiments'],
            'best_accuracy': summary['best_accuracy']['name'],
            'best_accuracy_value': summary['best_accuracy']['accuracy'],
            'smallest_model': summary['smallest_model']['name'],
            'smallest_model_size': summary['smallest_model']['model_size_mb'],
            'best_tradeoff': summary['best_tradeoff']['name'] if summary['best_tradeoff'] else None
        }, f, indent=2, ensure_ascii=False)
    
    print("\n=== 总结 ===")
    print(f"总实验数: {summary['total_experiments']}")
    print(f"最高准确率: {summary['best_accuracy']['name']} - {summary['best_accuracy']['accuracy']:.4f}")
    print(f"最小模型: {summary['smallest_model']['name']} - {summary['smallest_model']['model_size_mb']:.2f} MB")
    if summary['best_tradeoff']:
        print(f"最佳权衡: {summary['best_tradeoff']['name']}")
    
    print(f"\n所有图表已保存到: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='结果对比分析')
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='结果目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/comparison',
        help='输出目录'
    )
    
    args = parser.parse_args()
    main(args)
