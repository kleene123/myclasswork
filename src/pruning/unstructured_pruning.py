"""
非结构化剪枝实现

实现权重级别的稀疏化剪枝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List


class MagnitudePruner:
    """幅度剪枝器"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化幅度剪枝器
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        self.pruning_method = self.config.get('method', 'magnitude')
        self.global_pruning = self.config.get('global_pruning', True)
        
    def compute_weight_importance(
        self,
        weight: torch.Tensor,
        method: str = 'magnitude'
    ) -> torch.Tensor:
        """
        计算权重重要性
        
        Args:
            weight: 权重张量
            method: 计算方法 ('magnitude', 'l1', 'l2')
            
        Returns:
            重要性分数张量（与权重同shape）
        """
        if method == 'magnitude':
            return weight.abs()
        elif method == 'l1':
            return weight.abs()
        elif method == 'l2':
            return weight.pow(2)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def get_prunable_parameters(self) -> List[tuple]:
        """
        获取可剪枝的参数
        
        Returns:
            (name, parameter) 列表
        """
        prunable_params = []
        
        for name, param in self.model.named_parameters():
            # 只剪枝权重矩阵，跳过偏置和LayerNorm
            if 'weight' in name and param.dim() >= 2:
                if 'LayerNorm' not in name and 'norm' not in name:
                    prunable_params.append((name, param))
        
        return prunable_params
    
    def apply_global_pruning(
        self,
        sparsity: float
    ) -> Dict[str, any]:
        """
        应用全局剪枝（在所有参数中选择最不重要的权重）
        
        Args:
            sparsity: 目标稀疏度 (0-1)
            
        Returns:
            剪枝统计信息
        """
        prunable_params = self.get_prunable_parameters()
        
        # 收集所有权重的重要性分数
        all_scores = []
        for name, param in prunable_params:
            importance = self.compute_weight_importance(param.data, self.pruning_method)
            all_scores.append(importance.view(-1))
        
        # 合并所有分数
        all_scores = torch.cat(all_scores)
        
        # 计算阈值
        num_params = len(all_scores)
        num_to_prune = int(num_params * sparsity)
        threshold_idx = num_to_prune
        
        sorted_scores, _ = torch.sort(all_scores)
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else sorted_scores[-1]
        
        # 应用剪枝
        total_params = 0
        pruned_params = 0
        
        for name, param in prunable_params:
            importance = self.compute_weight_importance(param.data, self.pruning_method)
            mask = (importance > threshold).float()
            
            # 应用掩码
            param.data.mul_(mask)
            
            total_params += param.numel()
            pruned_params += (mask == 0).sum().item()
            
            print(f"{name}: {(mask == 0).sum().item() / param.numel() * 100:.2f}% 稀疏")
        
        actual_sparsity = pruned_params / total_params
        
        stats = {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'target_sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'threshold': threshold.item()
        }
        
        return stats
    
    def apply_layer_wise_pruning(
        self,
        sparsity: float
    ) -> Dict[str, any]:
        """
        应用逐层剪枝（每层独立剪枝）
        
        Args:
            sparsity: 目标稀疏度 (0-1)
            
        Returns:
            剪枝统计信息
        """
        prunable_params = self.get_prunable_parameters()
        
        total_params = 0
        pruned_params = 0
        layer_stats = {}
        
        for name, param in prunable_params:
            # 计算重要性
            importance = self.compute_weight_importance(param.data, self.pruning_method)
            
            # 计算阈值
            flat_importance = importance.view(-1)
            num_to_prune = int(len(flat_importance) * sparsity)
            
            if num_to_prune > 0:
                threshold_idx = num_to_prune
                sorted_importance, _ = torch.sort(flat_importance)
                threshold = sorted_importance[threshold_idx]
                
                # 创建掩码
                mask = (importance > threshold).float()
                
                # 应用掩码
                param.data.mul_(mask)
                
                layer_pruned = (mask == 0).sum().item()
                layer_total = param.numel()
                layer_sparsity = layer_pruned / layer_total
                
                total_params += layer_total
                pruned_params += layer_pruned
                
                layer_stats[name] = {
                    'params': layer_total,
                    'pruned': layer_pruned,
                    'sparsity': layer_sparsity
                }
                
                print(f"{name}: {layer_sparsity * 100:.2f}% 稀疏")
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        
        stats = {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'target_sparsity': sparsity,
            'actual_sparsity': actual_sparsity,
            'layer_stats': layer_stats
        }
        
        return stats


class UnstructuredPruning:
    """非结构化剪枝主类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化非结构化剪枝
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        self.pruner = MagnitudePruner(model, config)
        
    def apply_pruning(
        self,
        sparsity: float,
        global_pruning: bool = True
    ) -> Dict[str, any]:
        """
        应用非结构化剪枝
        
        Args:
            sparsity: 目标稀疏度
            global_pruning: 是否使用全局剪枝
            
        Returns:
            剪枝统计信息
        """
        print(f"\n=== 执行非结构化剪枝 (稀疏度: {sparsity:.2%}) ===")
        
        if global_pruning:
            print("使用全局剪枝策略...")
            stats = self.pruner.apply_global_pruning(sparsity)
        else:
            print("使用逐层剪枝策略...")
            stats = self.pruner.apply_layer_wise_pruning(sparsity)
        
        print(f"\n总参数: {stats['total_params']:,}")
        print(f"剪枝参数: {stats['pruned_params']:,}")
        print(f"实际稀疏度: {stats['actual_sparsity']:.2%}")
        
        return stats
    
    def get_sparsity(self) -> float:
        """
        获取当前模型的稀疏度
        
        Returns:
            稀疏度 (0-1)
        """
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            if param.dim() >= 2:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def remove_pruning_masks(self):
        """
        移除剪枝掩码，将零权重永久化
        
        注意：这会使权重真正变为零，之后无法恢复
        """
        for name, param in self.model.named_parameters():
            if param.dim() >= 2:
                # 权重已经被置零，这里只是确认
                pass
        
        print("剪枝掩码已移除，零权重已永久化")
