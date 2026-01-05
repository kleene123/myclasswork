#!/usr/bin/env python3
"""
综合对比实验

对比所有量化方法的效果，生成完整的对比分析结果
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
import argparse
from tqdm import tqdm

from src.models.bert_model import BERTModel
from src.utils.data_loader import load_imdb_dataset, create_dataloader
from src.evaluation.comparison import (
    compare_models,
    generate_comparison_table,
    generate_comparison_plots,
    export_results,
    generate_summary_report
)
from src.utils.logger import get_experiment_logger

logger = get_experiment_logger("comprehensive_comparison")


def load_all_models(results_dir: str, device: str = 'cpu'):
    """
    加载所有已训练和量化的模型
    
    Args:
        results_dir: 结果目录
        device: 设备
        
    Returns:
        models: 模型字典
    """
    logger.info("加载所有模型...")
    
    models = {}
    results_dir = Path(results_dir)
    
    # 加载基线模型
    baseline_path = results_dir / "baseline" / "model"
    if baseline_path.exists():
        logger.info("加载基线模型...")
        try:
            baseline_model = BERTModel.load(str(baseline_path))
            baseline_model.to(device)
            baseline_model.eval()
            models['基线模型'] = baseline_model.get_model()
            logger.info("✓ 基线模型加载成功")
        except Exception as e:
            logger.error(f"加载基线模型失败: {e}")
    
    # 加载动态量化模型
    dynamic_quant_path = results_dir / "dynamic_quant" / "quantized_model_full.pt"
    if dynamic_quant_path.exists():
        logger.info("加载动态量化模型...")
        try:
            dynamic_model = torch.load(dynamic_quant_path, map_location=device)
            dynamic_model.eval()
            models['动态量化'] = dynamic_model
            logger.info("✓ 动态量化模型加载成功")
        except Exception as e:
            logger.error(f"加载动态量化模型失败: {e}")
    
    # 加载静态量化模型
    static_quant_path = results_dir / "static_quant" / "quantized_model.pt"
    if static_quant_path.exists():
        logger.info("加载静态量化模型...")
        try:
            static_model = torch.load(static_quant_path, map_location=device)
            static_model.eval()
            models['静态量化'] = static_model
            logger.info("✓ 静态量化模型加载成功")
        except Exception as e:
            logger.error(f"加载静态量化模型失败: {e}")
    
    # 加载 QAT 模型
    qat_path = results_dir / "qat" / "quantized_model.pt"
    if qat_path.exists():
        logger.info("加载 QAT 模型...")
        try:
            qat_model = torch.load(qat_path, map_location=device)
            qat_model.eval()
            models['量化感知训练'] = qat_model
            logger.info("✓ QAT 模型加载成功")
        except Exception as e:
            logger.error(f"加载 QAT 模型失败: {e}")
    
    # 加载混合精度模型
    fp16_path = results_dir / "mixed_precision" / "fp16_model.pt"
    if fp16_path.exists():
        logger.info("加载 FP16 模型...")
        try:
            fp16_model = torch.load(fp16_path, map_location=device)
            fp16_model.eval()
            models['FP16混合精度'] = fp16_model
            logger.info("✓ FP16 模型加载成功")
        except Exception as e:
            logger.error(f"加载 FP16 模型失败: {e}")
    
    logger.info(f"成功加载 {len(models)} 个模型")
    return models


def run_comprehensive_comparison(
    config_path: str = "configs/experiment_config.yaml",
    results_dir: str = "results",
    device: str = None
):
    """
    运行综合对比实验
    
    Args:
        config_path: 配置文件路径
        results_dir: 结果目录
        device: 设备
    """
    logger.info("="*80)
    logger.info("开始综合对比实验")
    logger.info("="*80)
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if device is None:
        device = config.get('device', 'cpu')
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，使用 CPU")
        device = 'cpu'
    
    logger.info(f"使用设备: {device}")
    
    # 加载测试数据
    logger.info("加载测试数据...")
    eval_config = config.get('evaluation', {})
    test_samples = eval_config.get('test_samples', 1000)
    batch_size = eval_config.get('batch_size', 32)
    
    try:
        _, test_dataset, tokenizer = load_imdb_dataset(
            tokenizer_name="bert-base-uncased",
            max_length=128
        )
        
        # 限制测试样本数量
        if len(test_dataset) > test_samples:
            logger.info(f"限制测试样本数量为 {test_samples}")
            indices = list(range(test_samples))
            from torch.utils.data import Subset
            test_dataset = Subset(test_dataset, indices)
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        logger.info(f"测试数据加载完成: {len(test_dataset)} 样本")
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        logger.info("使用模拟数据进行演示...")
        # 创建模拟数据用于演示
        from torch.utils.data import TensorDataset, DataLoader
        dummy_input_ids = torch.randint(0, 1000, (test_samples, 128))
        dummy_labels = torch.randint(0, 2, (test_samples,))
        dummy_attention_mask = torch.ones(test_samples, 128)
        
        class DummyDataset:
            def __init__(self, input_ids, labels, attention_mask):
                self.input_ids = input_ids
                self.labels = labels
                self.attention_mask = attention_mask
            
            def __len__(self):
                return len(self.input_ids)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.input_ids[idx],
                    'labels': self.labels[idx],
                    'attention_mask': self.attention_mask[idx]
                }
        
        test_dataset = DummyDataset(dummy_input_ids, dummy_labels, dummy_attention_mask)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载所有模型
    models = load_all_models(results_dir, device)
    
    if not models:
        logger.error("没有找到任何模型，请先运行其他实验生成模型")
        return
    
    # 运行对比实验
    logger.info("\n" + "="*80)
    logger.info("开始对比评估...")
    logger.info("="*80 + "\n")
    
    num_iterations = eval_config.get('num_iterations', 100)
    
    try:
        results = compare_models(
            models=models,
            dataloader=test_loader,
            device=device,
            num_iterations=num_iterations,
            baseline_name='基线模型' if '基线模型' in models else None
        )
    except Exception as e:
        logger.error(f"对比评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存结果
    comparison_dir = Path(results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成对比表格
    logger.info("\n生成对比表格...")
    try:
        table_path = comparison_dir / "summary_table.csv"
        df = generate_comparison_table(results, save_path=str(table_path))
        logger.info(f"✓ 对比表格已保存: {table_path}")
        
        # 打印表格
        print("\n" + "="*80)
        print("实验结果对比表格")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
    except Exception as e:
        logger.error(f"生成对比表格失败: {e}")
    
    # 生成对比图表
    logger.info("生成对比图表...")
    try:
        generate_comparison_plots(results, save_dir=str(comparison_dir))
        logger.info(f"✓ 对比图表已保存: {comparison_dir}")
    except Exception as e:
        logger.error(f"生成对比图表失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 导出结果
    logger.info("导出实验结果...")
    try:
        export_results(results, save_dir=str(comparison_dir), format='csv')
        export_results(results, save_dir=str(comparison_dir), format='json')
        logger.info("✓ 结果已导出")
    except Exception as e:
        logger.error(f"导出结果失败: {e}")
    
    # 生成总结报告
    logger.info("生成总结报告...")
    try:
        report_path = comparison_dir / "summary_report.md"
        generate_summary_report(
            results,
            save_path=str(report_path),
            baseline_name='基线模型' if '基线模型' in models else None
        )
        logger.info(f"✓ 总结报告已保存: {report_path}")
    except Exception as e:
        logger.error(f"生成总结报告失败: {e}")
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("综合对比实验完成!")
    logger.info("="*80)
    logger.info(f"\n结果保存在: {comparison_dir}")
    logger.info("\n生成的文件:")
    logger.info("  - summary_table.csv         对比表格")
    logger.info("  - summary_report.md         总结报告")
    logger.info("  - accuracy_comparison.png   准确率对比图")
    logger.info("  - size_comparison.png       模型大小对比图")
    logger.info("  - speed_comparison.png      推理速度对比图")
    logger.info("  - radar_chart.png          综合性能雷达图")
    logger.info("  - accuracy_vs_size.png      准确率-大小权衡图")
    logger.info("  - summary_dashboard.png     综合仪表板")
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="综合对比实验")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='结果目录'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备 (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    run_comprehensive_comparison(
        config_path=args.config,
        results_dir=args.results_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
