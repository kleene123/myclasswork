"""
数据加载工具

支持 GLUE benchmark 和自定义文本分类数据集
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, List, Tuple
import numpy as np


class TextClassificationDataset(Dataset):
    """文本分类数据集类"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128
    ):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(
    dataset_name: str = "imdb",
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 ('imdb', 'sst2', 'ag_news' 等)
        tokenizer_name: 分词器名称
        max_length: 最大序列长度
        max_samples: 最大样本数（用于快速实验）
        cache_dir: 缓存目录
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    print(f"加载数据集: {dataset_name}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    
    # 加载数据集
    if dataset_name.lower() == "imdb":
        dataset = hf_load_dataset("imdb", cache_dir=cache_dir)
        
        # IMDB 只有 train 和 test，需要从 train 中分割出验证集
        train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_data = train_test_split['train']
        val_data = train_test_split['test']
        test_data = dataset['test']
        
        text_key = 'text'
        label_key = 'label'
        
    elif dataset_name.lower() == "sst2":
        dataset = hf_load_dataset("glue", "sst2", cache_dir=cache_dir)
        
        train_data = dataset['train']
        val_data = dataset['validation']
        test_data = dataset['validation']  # SST-2 测试集没有标签
        
        text_key = 'sentence'
        label_key = 'label'
        
    elif dataset_name.lower() == "ag_news":
        dataset = hf_load_dataset("ag_news", cache_dir=cache_dir)
        
        train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_data = train_test_split['train']
        val_data = train_test_split['test']
        test_data = dataset['test']
        
        text_key = 'text'
        label_key = 'label'
        
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 限制样本数（用于快速实验）
    if max_samples is not None:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 5, len(val_data))))
        test_data = test_data.select(range(min(max_samples // 5, len(test_data))))
    
    # 创建数据集
    train_dataset = TextClassificationDataset(
        texts=train_data[text_key],
        labels=train_data[label_key],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = TextClassificationDataset(
        texts=val_data[text_key],
        labels=val_data[label_key],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = TextClassificationDataset(
        texts=test_data[text_key],
        labels=test_data[label_key],
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> TorchDataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        DataLoader
    """
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class DataLoader:
    """数据加载器管理类"""
    
    def __init__(self, config: Dict):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.dataset_name = config.get('dataset', 'imdb')
        self.batch_size = config.get('batch_size', 32)
        self.max_length = config.get('max_length', 128)
        self.max_samples = config.get('max_samples', None)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def prepare_data(
        self,
        tokenizer_name: str = "bert-base-uncased",
        cache_dir: Optional[str] = None
    ):
        """
        准备数据
        
        Args:
            tokenizer_name: 分词器名称
            cache_dir: 缓存目录
        """
        # 加载数据集
        self.train_dataset, self.val_dataset, self.test_dataset = load_dataset(
            dataset_name=self.dataset_name,
            tokenizer_name=tokenizer_name,
            max_length=self.max_length,
            max_samples=self.max_samples,
            cache_dir=cache_dir
        )
        
        # 创建数据加载器
        self.train_loader = create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_loader = create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        self.test_loader = create_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        print("数据加载器准备完成")
        
    def get_train_loader(self) -> TorchDataLoader:
        """获取训练数据加载器"""
        return self.train_loader
    
    def get_val_loader(self) -> TorchDataLoader:
        """获取验证数据加载器"""
        return self.val_loader
    
    def get_test_loader(self) -> TorchDataLoader:
        """获取测试数据加载器"""
        return self.test_loader
