"""
BERT 模型封装

提供 BERT 模型的加载、训练、推理功能
"""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig
)
from typing import Optional, Dict, Any
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BERTModel:
    """BERT 模型封装类"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        cache_dir: Optional[str] = "./models/cache"
    ):
        """
        初始化 BERT 模型
        
        Args:
            model_name: 预训练模型名称
            num_labels: 分类标签数量
            cache_dir: 缓存目录
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.cache_dir = cache_dir
        
        logger.info(f"加载 BERT 模型: {model_name}")
        
        # 加载配置
        self.config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir
        )
        
        # 加载模型
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir
        )
        
        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("BERT 模型加载完成")
    
    def get_model(self) -> nn.Module:
        """获取模型对象"""
        return self.model
    
    def get_tokenizer(self):
        """获取 tokenizer"""
        return self.tokenizer
    
    def to(self, device: str):
        """移动模型到指定设备"""
        self.model.to(device)
        return self
    
    def eval(self):
        """设置为评估模式"""
        self.model.eval()
        return self
    
    def train(self):
        """设置为训练模式"""
        self.model.train()
        return self
    
    def save(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        logger.info(f"保存模型到: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def load(cls, load_path: str, num_labels: int = 2):
        """
        从路径加载模型
        
        Args:
            load_path: 加载路径
            num_labels: 标签数量
            
        Returns:
            bert_model: BERT 模型实例
        """
        logger.info(f"从路径加载模型: {load_path}")
        
        instance = cls.__new__(cls)
        instance.num_labels = num_labels
        instance.model_name = load_path
        
        instance.model = BertForSequenceClassification.from_pretrained(load_path)
        instance.tokenizer = BertTokenizer.from_pretrained(load_path)
        instance.config = instance.model.config
        
        return instance
    
    def predict(
        self,
        texts: list,
        batch_size: int = 32,
        max_length: int = 128,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        批量预测
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            max_length: 最大长度
            device: 设备
            
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        self.model.to(device)
        
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 编码
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                # 移动到设备
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                
                # 预测
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions)
