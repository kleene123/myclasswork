#!/usr/bin/env python3
"""
运行所有实验

自动运行所有量化实验并生成完整报告
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import subprocess
from src.utils.logger import get_experiment_logger

logger = get_experiment_logger("run_all_experiments")


def run_experiment(script_name: str, description: str):
    """
    运行单个实验脚本
    
    Args:
        script_name: 脚本名称
        description: 实验描述
    """
    logger.info("="*80)
    logger.info(f"运行实验: {description}")
    logger.info(f"脚本: {script_name}")
    logger.info("="*80)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.warning(f"实验脚本不存在: {script_path}")
        logger.info(f"跳过实验: {description}")
        return False
    
    try:
        # 运行脚本
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 打印输出
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            logger.info(f"✓ 实验完成: {description}\n")
            return True
        else:
            logger.error(f"✗ 实验失败: {description}")
            if result.stderr:
                logger.error(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ 实验超时: {description}")
        return False
    except Exception as e:
        logger.error(f"✗ 实验异常: {description}")
        logger.error(f"错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="运行所有量化实验")
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='跳过基线模型训练（如果已有模型）'
    )
    parser.add_argument(
        '--only-comparison',
        action='store_true',
        help='只运行综合对比实验'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("Transformer 模型量化实验 - 自动运行所有实验")
    logger.info("="*80 + "\n")
    
    experiments = []
    
    if not args.only_comparison:
        # 定义所有实验
        experiments = [
            ("01_baseline_training.py", "实验1: 训练基线模型"),
            ("02_dynamic_quantization.py", "实验2: 动态量化"),
            ("03_static_quantization.py", "实验3: 静态量化"),
            ("04_qat_experiment.py", "实验4: 量化感知训练"),
            ("05_mixed_precision.py", "实验5: 混合精度"),
        ]
        
        if args.skip_training:
            logger.info("跳过基线模型训练")
            experiments = experiments[1:]
    
    # 总是运行综合对比
    experiments.append(("06_comprehensive_comparison.py", "实验6: 综合对比"))
    
    # 记录结果
    results = {}
    
    # 运行所有实验
    for script, description in experiments:
        success = run_experiment(script, description)
        results[description] = "成功" if success else "失败"
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("实验运行总结")
    logger.info("="*80)
    
    for exp, status in results.items():
        status_icon = "✓" if status == "成功" else "✗"
        logger.info(f"{status_icon} {exp}: {status}")
    
    logger.info("="*80 + "\n")
    
    # 统计
    total = len(results)
    success_count = sum(1 for s in results.values() if s == "成功")
    
    logger.info(f"总实验数: {total}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {total - success_count}")
    
    if success_count == total:
        logger.info("\n🎉 所有实验成功完成！")
        logger.info("\n查看结果:")
        logger.info("  - 对比图表: results/comparison/")
        logger.info("  - 课程设计报告: report/课程设计报告.md")
    else:
        logger.warning("\n⚠️ 部分实验失败，请检查日志")
    
    logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
