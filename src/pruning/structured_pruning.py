"""
结构化剪枝实现

包括注意力头剪枝和 FFN 剪枝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import BertForSequenceClassification


class AttentionHeadPruner:
    """注意力头剪枝器"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化注意力头剪枝器
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        self.pruning_method = self.config.get('pruning_method', 'importance')
        
    def compute_head_importance(
        self,
        dataloader,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        计算注意力头重要性分数
        
        Args:
            dataloader: 数据加载器
            num_samples: 用于计算的样本数量
            
        Returns:
            重要性分数张量 (num_layers, num_heads)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 获取模型配置
        if hasattr(self.model, 'config'):
            num_layers = self.model.config.num_hidden_layers
            num_heads = self.model.config.num_attention_heads
        else:
            # 对于自定义模型
            num_layers = self.model.num_layers if hasattr(self.model, 'num_layers') else 12
            num_heads = 12  # 默认值
        
        head_importance = torch.zeros(num_layers, num_heads).to(device)
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if total_samples >= num_samples:
                    break
                
                # 准备输入
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 前向传播获取注意力权重
                if hasattr(self.model, 'model'):
                    # BERTWrapper
                    outputs = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                
                # 获取注意力权重
                attentions = outputs.attentions if hasattr(outputs, 'attentions') else outputs[1]
                
                # 累加每个头的注意力权重
                for layer_idx, attention in enumerate(attentions):
                    # attention shape: (batch_size, num_heads, seq_len, seq_len)
                    head_importance[layer_idx] += attention.abs().sum(dim=(0, 2, 3))
                
                total_samples += input_ids.size(0)
        
        # 归一化
        if total_samples > 0:
            head_importance = head_importance / total_samples
        
        return head_importance
    
    def select_heads_to_prune(
        self,
        head_importance: torch.Tensor,
        num_heads_to_prune: int
    ) -> List[Tuple[int, int]]:
        """
        选择要剪枝的注意力头
        
        Args:
            head_importance: 头重要性分数
            num_heads_to_prune: 要剪枝的头数量
            
        Returns:
            要剪枝的头列表 [(layer_idx, head_idx), ...]
        """
        # 展平重要性分数
        num_layers, num_heads = head_importance.shape
        flat_importance = head_importance.view(-1)
        
        # 找到重要性最低的头
        _, indices = torch.sort(flat_importance)
        indices_to_prune = indices[:num_heads_to_prune]
        
        # 转换为 (layer, head) 格式
        heads_to_prune = []
        for idx in indices_to_prune:
            layer_idx = idx.item() // num_heads
            head_idx = idx.item() % num_heads
            heads_to_prune.append((layer_idx, head_idx))
        
        return heads_to_prune
    
    def prune_heads(
        self,
        heads_to_prune: Dict[int, List[int]]
    ):
        """
        执行注意力头剪枝
        
        Args:
            heads_to_prune: 要剪枝的头字典 {layer_idx: [head_indices]}
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'bert'):
            # BERTWrapper with BertForSequenceClassification
            for layer_idx, head_indices in heads_to_prune.items():
                if layer_idx < len(self.model.model.bert.encoder.layer):
                    layer = self.model.model.bert.encoder.layer[layer_idx]
                    layer.attention.prune_heads(head_indices)
        elif hasattr(self.model, 'encoder_layers'):
            # 自定义 Transformer 模型
            print("注意: 自定义模型的头剪枝需要手动实现")
    
    def apply_pruning(
        self,
        dataloader,
        num_heads_to_prune: int,
        num_samples: int = 100
    ) -> Dict[str, any]:
        """
        应用注意力头剪枝
        
        Args:
            dataloader: 数据加载器
            num_heads_to_prune: 要剪枝的头总数
            num_samples: 用于计算重要性的样本数
            
        Returns:
            剪枝统计信息
        """
        # 计算头重要性
        print("计算注意力头重要性...")
        head_importance = self.compute_head_importance(dataloader, num_samples)
        
        # 选择要剪枝的头
        print(f"选择 {num_heads_to_prune} 个头进行剪枝...")
        heads_to_prune_list = self.select_heads_to_prune(head_importance, num_heads_to_prune)
        
        # 转换为字典格式
        heads_to_prune_dict = {}
        for layer_idx, head_idx in heads_to_prune_list:
            if layer_idx not in heads_to_prune_dict:
                heads_to_prune_dict[layer_idx] = []
            heads_to_prune_dict[layer_idx].append(head_idx)
        
        # 执行剪枝
        print("执行剪枝...")
        self.prune_heads(heads_to_prune_dict)
        
        stats = {
            'heads_pruned': len(heads_to_prune_list),
            'pruned_heads': heads_to_prune_list,
            'head_importance': head_importance.cpu().numpy()
        }
        
        return stats


class FFNPruner:
    """前馈网络剪枝器"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化 FFN 剪枝器
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        self.pruning_method = self.config.get('pruning_method', 'magnitude')
        
    def compute_neuron_importance(
        self,
        layer_idx: int
    ) -> torch.Tensor:
        """
        计算FFN中间层神经元的重要性
        
        Args:
            layer_idx: 层索引
            
        Returns:
            神经元重要性分数
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'bert'):
            layer = self.model.model.bert.encoder.layer[layer_idx]
            intermediate_weight = layer.intermediate.dense.weight.data
        elif hasattr(self.model, 'encoder_layers'):
            layer = self.model.encoder_layers[layer_idx]
            intermediate_weight = layer.feed_forward.linear1.weight.data
        else:
            raise ValueError("不支持的模型类型")
        
        # 使用L1范数作为重要性度量
        if self.pruning_method == 'magnitude':
            importance = torch.norm(intermediate_weight, p=1, dim=1)
        else:
            importance = torch.norm(intermediate_weight, p=2, dim=1)
        
        return importance
    
    def prune_ffn_layer(
        self,
        layer_idx: int,
        sparsity: float
    ):
        """
        剪枝特定层的FFN
        
        Args:
            layer_idx: 层索引
            sparsity: 稀疏度 (0-1)
        """
        importance = self.compute_neuron_importance(layer_idx)
        num_neurons = len(importance)
        num_to_prune = int(num_neurons * sparsity)
        
        # 选择要剪枝的神经元
        _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)
        
        # 创建掩码
        mask = torch.ones_like(importance)
        mask[indices_to_prune] = 0
        
        # 应用掩码到权重
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'bert'):
            layer = self.model.model.bert.encoder.layer[layer_idx]
            layer.intermediate.dense.weight.data *= mask.unsqueeze(1)
            layer.output.dense.weight.data *= mask.unsqueeze(0)
        elif hasattr(self.model, 'encoder_layers'):
            layer = self.model.encoder_layers[layer_idx]
            layer.feed_forward.linear1.weight.data *= mask.unsqueeze(1)
            layer.feed_forward.linear2.weight.data *= mask.unsqueeze(0)
        
        return num_to_prune
    
    def apply_pruning(
        self,
        sparsity: float
    ) -> Dict[str, any]:
        """
        应用FFN剪枝到所有层
        
        Args:
            sparsity: 目标稀疏度
            
        Returns:
            剪枝统计信息
        """
        total_neurons_pruned = 0
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'bert'):
            num_layers = len(self.model.model.bert.encoder.layer)
        elif hasattr(self.model, 'encoder_layers'):
            num_layers = len(self.model.encoder_layers)
        else:
            raise ValueError("不支持的模型类型")
        
        for layer_idx in range(num_layers):
            num_pruned = self.prune_ffn_layer(layer_idx, sparsity)
            total_neurons_pruned += num_pruned
            print(f"层 {layer_idx}: 剪枝了 {num_pruned} 个神经元")
        
        stats = {
            'total_neurons_pruned': total_neurons_pruned,
            'sparsity': sparsity,
            'num_layers': num_layers
        }
        
        return stats


class StructuredPruning:
    """结构化剪枝主类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化结构化剪枝
        
        Args:
            model: 要剪枝的模型
            config: 剪枝配置
        """
        self.model = model
        self.config = config or {}
        
        self.head_pruner = AttentionHeadPruner(model, config.get('attention_head_pruning', {}))
        self.ffn_pruner = FFNPruner(model, config.get('ffn_pruning', {}))
        
    def apply_pruning(
        self,
        dataloader=None,
        prune_heads: bool = True,
        prune_ffn: bool = True,
        num_heads_to_prune: int = 0,
        ffn_sparsity: float = 0.0
    ) -> Dict[str, any]:
        """
        应用结构化剪枝
        
        Args:
            dataloader: 数据加载器（用于计算头重要性）
            prune_heads: 是否剪枝注意力头
            prune_ffn: 是否剪枝FFN
            num_heads_to_prune: 要剪枝的头数量
            ffn_sparsity: FFN稀疏度
            
        Returns:
            剪枝统计信息
        """
        stats = {}
        
        if prune_heads and num_heads_to_prune > 0:
            if dataloader is None:
                raise ValueError("剪枝注意力头需要提供 dataloader")
            print("\n=== 执行注意力头剪枝 ===")
            head_stats = self.head_pruner.apply_pruning(dataloader, num_heads_to_prune)
            stats['attention_heads'] = head_stats
        
        if prune_ffn and ffn_sparsity > 0:
            print("\n=== 执行 FFN 剪枝 ===")
            ffn_stats = self.ffn_pruner.apply_pruning(ffn_sparsity)
            stats['ffn'] = ffn_stats
        
        return stats
