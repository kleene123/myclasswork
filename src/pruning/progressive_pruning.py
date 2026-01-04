"""
渐进式剪枝实现

实现迭代式的渐进剪枝策略
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from .unstructured_pruning import MagnitudePruner


class ProgressivePruning:
    """渐进式剪枝类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化渐进式剪枝
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        
        self.initial_sparsity = self.config.get('initial_sparsity', 0.0)
        self.final_sparsity = self.config.get('final_sparsity', 0.5)
        self.num_iterations = self.config.get('num_iterations', 5)
        self.pruning_frequency = self.config.get('pruning_frequency', 100)
        
        self.current_iteration = 0
        self.current_sparsity = self.initial_sparsity
        
        self.pruner = MagnitudePruner(model, config)
        
    def compute_target_sparsity(self, iteration: int) -> float:
        """
        计算当前迭代的目标稀疏度
        
        使用多项式衰减策略
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            目标稀疏度
        """
        if iteration >= self.num_iterations:
            return self.final_sparsity
        
        # 使用三次多项式衰减
        progress = iteration / self.num_iterations
        sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * (
            3 * progress ** 2 - 2 * progress ** 3
        )
        
        return sparsity
    
    def prune_iteration(
        self,
        iteration: int,
        global_pruning: bool = True
    ) -> Dict[str, any]:
        """
        执行一次剪枝迭代
        
        Args:
            iteration: 迭代次数
            global_pruning: 是否使用全局剪枝
            
        Returns:
            剪枝统计信息
        """
        target_sparsity = self.compute_target_sparsity(iteration)
        
        print(f"\n=== 渐进式剪枝迭代 {iteration + 1}/{self.num_iterations} ===")
        print(f"目标稀疏度: {target_sparsity:.2%}")
        
        if global_pruning:
            stats = self.pruner.apply_global_pruning(target_sparsity)
        else:
            stats = self.pruner.apply_layer_wise_pruning(target_sparsity)
        
        self.current_iteration = iteration
        self.current_sparsity = stats['actual_sparsity']
        
        stats['iteration'] = iteration
        stats['target_sparsity'] = target_sparsity
        
        return stats
    
    def apply_progressive_pruning(
        self,
        trainer=None,
        eval_dataloader=None,
        fine_tune_fn: Optional[Callable] = None,
        global_pruning: bool = True
    ) -> Dict[str, any]:
        """
        应用完整的渐进式剪枝流程
        
        Args:
            trainer: 训练器（用于微调）
            eval_dataloader: 评估数据加载器
            fine_tune_fn: 微调函数
            global_pruning: 是否使用全局剪枝
            
        Returns:
            剪枝历史统计
        """
        history = {
            'iterations': [],
            'sparsities': [],
            'metrics': []
        }
        
        for iteration in range(self.num_iterations):
            # 执行剪枝
            stats = self.prune_iteration(iteration, global_pruning)
            
            # 记录统计
            history['iterations'].append(iteration)
            history['sparsities'].append(stats['actual_sparsity'])
            
            # 微调（如果提供）
            if fine_tune_fn is not None:
                print(f"\n微调模型（迭代 {iteration + 1}）...")
                metrics = fine_tune_fn(self.model, iteration)
                history['metrics'].append(metrics)
            elif trainer is not None and eval_dataloader is not None:
                # 使用提供的trainer进行简单评估
                print(f"\n评估模型（迭代 {iteration + 1}）...")
                metrics = trainer.evaluate(eval_dataloader)
                history['metrics'].append(metrics)
            
            print(f"迭代 {iteration + 1} 完成")
            print(f"当前稀疏度: {self.current_sparsity:.2%}")
        
        print("\n=== 渐进式剪枝完成 ===")
        print(f"最终稀疏度: {self.current_sparsity:.2%}")
        
        return history
    
    def get_pruning_schedule(self) -> Dict[int, float]:
        """
        获取剪枝计划
        
        Returns:
            {iteration: target_sparsity} 字典
        """
        schedule = {}
        for i in range(self.num_iterations):
            schedule[i] = self.compute_target_sparsity(i)
        return schedule
    
    def reset(self):
        """重置剪枝状态"""
        self.current_iteration = 0
        self.current_sparsity = self.initial_sparsity
        
        # 注意：这不会恢复模型权重
        print("剪枝状态已重置（模型权重未恢复）")
