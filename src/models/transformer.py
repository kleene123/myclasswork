"""
Transformer 模型实现

简化的 Transformer 模型实现，用于教学和实验目的
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            dropout: Dropout 概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 用于剪枝的注意力头掩码
        self.head_mask = None
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            query: 查询张量 (batch_size, seq_len, d_model)
            key: 键张量 (batch_size, seq_len, d_model)
            value: 值张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            输出张量和可选的注意力权重
        """
        batch_size = query.size(0)
        
        # 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力头掩码（用于剪枝）
        if self.head_mask is not None:
            attention_weights = attention_weights * self.head_mask
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.W_o(context)
        
        if return_attention:
            return output, attention_weights
        return output, None


class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络中间层维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播"""
        # 自注意力子层
        attention_output, attention_weights = self.self_attention(x, x, x, mask, return_attention)
        x = self.norm1(x + self.dropout1(attention_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attention_weights


class TransformerModel(nn.Module):
    """完整的 Transformer 模型（仅编码器）"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数量
            num_layers: 编码器层数
            d_ff: 前馈网络维度
            max_seq_length: 最大序列长度
            num_classes: 分类类别数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(
            self._get_positional_encoding(max_seq_length, d_model),
            requires_grad=False
        )
        
        # Transformer 编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _get_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """生成位置编码"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            return_attention: 是否返回注意力权重
            
        Returns:
            分类logits和可选的注意力权重列表
        """
        seq_len = input_ids.size(1)
        
        # 嵌入 + 位置编码
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 准备注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 通过编码器层
        all_attention_weights = [] if return_attention else None
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, attention_mask, return_attention)
            if return_attention:
                all_attention_weights.append(attention_weights)
        
        # 使用 [CLS] token 的表示进行分类（这里使用第一个token）
        cls_representation = x[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_representation)
        
        return logits, all_attention_weights
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        获取模型参数数量
        
        Args:
            trainable_only: 是否只统计可训练参数
            
        Returns:
            参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
