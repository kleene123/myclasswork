"""
数据加载工具

提供数据集加载和预处理功能
"""

from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TextClassificationDataset(Dataset):
    """文本分类数据集封装"""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def load_imdb_dataset(
    tokenizer_name: str = "bert-base-uncased",
    max_length: int = 128,
    cache_dir: str = "./data/cache"
) -> Tuple[Dataset, Dataset, AutoTokenizer]:
    """
    加载 IMDB 数据集
    
    Args:
        tokenizer_name: tokenizer 名称
        max_length: 最大序列长度
        cache_dir: 缓存目录
        
    Returns:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        tokenizer: tokenizer
    """
    logger.info(f"加载 IMDB 数据集，tokenizer: {tokenizer_name}")
    
    # 加载数据集
    dataset = load_dataset("imdb", cache_dir=cache_dir)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    
    # 编码函数
    def encode_batch(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
    
    # 编码数据集
    logger.info("编码训练集...")
    train_encodings = encode_batch(dataset['train'])
    train_labels = dataset['train']['label']
    
    logger.info("编码测试集...")
    test_encodings = encode_batch(dataset['test'])
    test_labels = dataset['test']['label']
    
    # 创建 Dataset 对象
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    test_dataset = TextClassificationDataset(test_encodings, test_labels)
    
    logger.info(f"数据集加载完成: 训练集 {len(train_dataset)} 样本，测试集 {len(test_dataset)} 样本")
    
    return train_dataset, test_dataset, tokenizer


def create_calibration_dataloader(
    dataset: Dataset,
    num_samples: int = 1000,
    batch_size: int = 32,
    seed: int = 42
) -> DataLoader:
    """
    创建校准数据加载器（用于静态量化）
    
    Args:
        dataset: 数据集
        num_samples: 校准样本数量
        batch_size: 批次大小
        seed: 随机种子
        
    Returns:
        calibration_loader: 校准数据加载器
    """
    # 随机选择样本
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    calibration_dataset = Subset(dataset, indices)
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"创建校准数据加载器: {len(calibration_dataset)} 样本，批次大小 {batch_size}")
    
    return calibration_loader


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        dataloader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
