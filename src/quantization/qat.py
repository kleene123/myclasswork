"""
量化感知训练 (Quantization-Aware Training) 实现
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Optional
from torch.utils.data import DataLoader


class QuantizationAwareTraining:
    """量化感知训练类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化量化感知训练
        
        Args:
            model: 要量化的模型
            config: 量化配置
        """
        self.model = model
        self.config = config or {}
        
        self.dtype = self.config.get('dtype', 'int8')
        self.num_calibration_batches = self.config.get('num_calibration_batches', 32)
        self.freeze_bn_stats = self.config.get('freeze_bn_stats', False)
        
        self.qat_model = None
        
    def prepare_model_for_qat(self):
        """
        准备模型进行量化感知训练
        
        插入伪量化节点
        """
        # 设置为训练模式
        self.model.train()
        
        # 配置量化
        if self.dtype == 'int8':
            self.model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        else:
            self.model.qconfig = quant.default_qat_qconfig
        
        # 准备QAT（插入伪量化节点）
        if hasattr(self.model, 'model'):
            # BERTWrapper
            self.qat_model = quant.prepare_qat(self.model.model, inplace=False)
            self.model.model = self.qat_model
        else:
            self.qat_model = quant.prepare_qat(self.model, inplace=False)
            self.model = self.qat_model
        
        print("模型已准备好进行量化感知训练")
        print("伪量化节点已插入")
        
        return self.model
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ) -> float:
        """
        单步训练
        
        Args:
            batch: 训练批次
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            
        Returns:
            损失值
        """
        self.model.train()
        
        # 准备输入
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        labels = batch['labels'].to(device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        if hasattr(self.model, 'model'):
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
        optimizer.step()
        
        return loss.item()
    
    def train_qat(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 3,
        device: torch.device = None
    ) -> Dict[str, list]:
        """
        执行量化感知训练
        
        Args:
            train_dataloader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            num_epochs: 训练轮数
            device: 设备
            
        Returns:
            训练历史
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        history = {
            'epoch': [],
            'loss': []
        }
        
        print(f"开始量化感知训练，共 {num_epochs} 轮...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                loss = self.train_step(batch, optimizer, criterion, device)
                epoch_loss += loss
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            history['epoch'].append(epoch)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
            
            # 冻结BN统计（如果配置）
            if self.freeze_bn_stats and epoch == 0:
                print("冻结 BatchNorm 统计...")
                if hasattr(self.model, 'model'):
                    self.model.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                else:
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        print("量化感知训练完成")
        
        return history
    
    def convert_to_quantized(self):
        """
        将QAT模型转换为量化模型
        
        Returns:
            量化后的模型
        """
        print("转换QAT模型为量化格式...")
        
        # 设置为评估模式
        self.model.eval()
        
        if hasattr(self.model, 'model'):
            # BERTWrapper
            quantized_model = quant.convert(self.model.model, inplace=False)
            self.model.model = quantized_model
        else:
            quantized_model = quant.convert(self.model, inplace=False)
            self.model = quantized_model
        
        print("QAT模型已转换为量化格式")
        
        return self.model
    
    def apply_qat(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int = 3,
        device: torch.device = None
    ):
        """
        完整的QAT流程
        
        Args:
            train_dataloader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            num_epochs: 训练轮数
            device: 设备
            
        Returns:
            量化后的模型和训练历史
        """
        # 准备模型
        self.prepare_model_for_qat()
        
        # 训练
        history = self.train_qat(
            train_dataloader,
            optimizer,
            criterion,
            num_epochs,
            device
        )
        
        # 转换为量化模型
        quantized_model = self.convert_to_quantized()
        
        return quantized_model, history
    
    def get_model_size(self) -> float:
        """
        获取模型大小（MB）
        
        Returns:
            模型大小（MB）
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return size_mb
