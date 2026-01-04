"""
训练器模块

用于模型训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm
import os


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
            device: 设备
        """
        self.model = model
        self.config = config or {}
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 训练参数
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.epochs = self.config.get('epochs', 3)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.warmup_steps = self.config.get('warmup_steps', 0)
        self.logging_steps = self.config.get('logging_steps', 100)
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
        # 训练历史
        self.history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
    def setup_optimizer(self, optimizer_type: str = 'adamw'):
        """
        设置优化器
        
        Args:
            optimizer_type: 优化器类型
        """
        if optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
    def setup_scheduler(self, num_training_steps: int):
        """
        设置学习率调度器
        
        Args:
            num_training_steps: 总训练步数
        """
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        if self.warmup_steps > 0:
            # 使用warmup
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )
            
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps - self.warmup_steps
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_steps]
            )
        else:
            # 线性衰减
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        epoch: int
    ) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            criterion: 损失函数
            epoch: 当前epoch
            
        Returns:
            平均损失
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备输入
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            labels = batch['labels'].to(self.device)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'model'):
                # BERTWrapper
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            # 日志
            if batch_idx % self.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        evaluator=None
    ) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            evaluator: 评估器
            
        Returns:
            训练历史
        """
        # 设置损失函数
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 设置优化器
        if self.optimizer is None:
            self.setup_optimizer()
        
        # 设置调度器
        num_training_steps = len(train_loader) * self.epochs
        if self.scheduler is None:
            self.setup_scheduler(num_training_steps)
        
        print(f"开始训练，共 {self.epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"学习率: {self.learning_rate}")
        
        # 训练循环
        for epoch in range(self.epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, criterion, epoch)
            
            # 验证
            if val_loader is not None and evaluator is not None:
                val_metrics = evaluator.evaluate(val_loader)
                val_loss = val_metrics.get('loss', 0)
                val_accuracy = val_metrics.get('accuracy', 0)
                
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                print(f"训练损失: {train_loss:.4f}")
                print(f"验证损失: {val_loss:.4f}")
                print(f"验证准确率: {val_accuracy:.4f}")
                
                # 记录历史
                self.history['epoch'].append(epoch)
                self.history['loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            else:
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                print(f"训练损失: {train_loss:.4f}")
                
                self.history['epoch'].append(epoch)
                self.history['loss'].append(train_loss)
        
        print("\n训练完成!")
        
        return self.history
    
    def save_checkpoint(self, save_path: str, epoch: int):
        """
        保存检查点
        
        Args:
            save_path: 保存路径
            epoch: 当前epoch
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"检查点已保存到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        print(f"检查点已从 {checkpoint_path} 加载")
        print(f"继续从 epoch {checkpoint['epoch']} 训练")
