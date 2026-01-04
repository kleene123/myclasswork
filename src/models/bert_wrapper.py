"""
BERT 模型封装

使用 Hugging Face Transformers 库的 BERT 模型
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForSequenceClassification
from typing import Optional, Dict, Any


class BERTWrapper(nn.Module):
    """BERT 模型封装类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 BERT 模型
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类标签数量
            config: 额外的配置参数
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # 加载预训练模型配置
        if config:
            bert_config = BertConfig.from_pretrained(model_name)
            for key, value in config.items():
                if hasattr(bert_config, key):
                    setattr(bert_config, key, value)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, 
                config=bert_config,
                num_labels=num_labels
            )
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
        
        self.config = self.model.config
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: Token类型IDs
            labels: 标签（用于计算损失）
            return_dict: 是否返回字典格式
            
        Returns:
            模型输出
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=return_dict,
            output_attentions=True,
            output_hidden_states=True
        )
        
        return outputs
    
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
    
    def get_model_size(self) -> float:
        """
        获取模型大小（MB）
        
        Returns:
            模型大小（MB）
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def get_attention_heads_importance(
        self,
        dataloader,
        num_samples: int = 100
    ) -> torch.Tensor:
        """
        计算注意力头的重要性分数
        
        Args:
            dataloader: 数据加载器
            num_samples: 用于计算的样本数量
            
        Returns:
            注意力头重要性分数张量 (num_layers, num_heads)
        """
        self.eval()
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        
        head_importance = torch.zeros(num_layers, num_heads).to(self.model.device)
        
        samples_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 计算注意力权重的平均值作为重要性指标
                attentions = outputs.attentions
                for layer_idx, attention in enumerate(attentions):
                    # attention shape: (batch_size, num_heads, seq_len, seq_len)
                    head_importance[layer_idx] += attention.abs().sum(dim=(0, 2, 3))
                
                samples_processed += input_ids.size(0)
        
        # 归一化
        head_importance = head_importance / samples_processed
        
        return head_importance
    
    def save_pretrained(self, save_directory: str):
        """
        保存模型
        
        Args:
            save_directory: 保存目录
        """
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        从本地路径加载模型
        
        Args:
            model_path: 模型路径
            **kwargs: 其他参数
            
        Returns:
            BERTWrapper 实例
        """
        wrapper = cls.__new__(cls)
        super(BERTWrapper, wrapper).__init__()
        
        wrapper.model = BertForSequenceClassification.from_pretrained(model_path, **kwargs)
        wrapper.config = wrapper.model.config
        wrapper.model_name = model_path
        wrapper.num_labels = wrapper.config.num_labels
        
        return wrapper
