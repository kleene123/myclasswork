"""
评估器模块

用于模型评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from ..utils.metrics import compute_metrics


class Evaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 设备
        """
        self.model = model
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        return_predictions: bool = False
    ) -> Dict[str, any]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            return_predictions: 是否返回预测结果
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        num_batches = 0
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估中"):
                # 准备输入
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels'].to(self.device)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 前向传播
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
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和标签
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # 计算指标
        metrics = compute_metrics(all_predictions, all_labels)
        
        # 添加损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics['loss'] = avg_loss
        
        # 打印结果
        print(f"\n=== 评估结果 ===")
        print(f"损失: {metrics['loss']:.4f}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        
        # 是否返回预测
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
            metrics['logits'] = all_logits
        
        return metrics
    
    def predict(
        self,
        dataloader: DataLoader,
        return_logits: bool = False
    ) -> np.ndarray:
        """
        进行预测
        
        Args:
            dataloader: 数据加载器
            return_logits: 是否返回logits
            
        Returns:
            预测结果或logits
        """
        self.model.eval()
        
        all_predictions = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
                # 准备输入
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # 前向传播
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
                
                # 收集预测
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        if return_logits:
            return np.array(all_logits)
        
        return np.array(all_predictions)
    
    def evaluate_detailed(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, any]:
        """
        详细评估（包含每个类别的指标）
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            
        Returns:
            详细评估指标
        """
        # 基础评估
        metrics = self.evaluate(dataloader, criterion, return_predictions=True)
        
        predictions = metrics.pop('predictions')
        labels = metrics.pop('labels')
        logits = metrics.pop('logits')
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(labels, predictions)
        
        # 分类报告
        report = classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0
        )
        
        metrics['confusion_matrix'] = cm
        metrics['classification_report'] = report
        
        print("\n=== 分类报告 ===")
        print(classification_report(labels, predictions, zero_division=0))
        
        return metrics
    
    def compare_models(
        self,
        other_model: nn.Module,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None
    ) -> Dict[str, any]:
        """
        比较两个模型的性能
        
        Args:
            other_model: 另一个模型
            dataloader: 数据加载器
            criterion: 损失函数
            
        Returns:
            比较结果
        """
        # 评估当前模型
        print("评估模型1...")
        metrics1 = self.evaluate(dataloader, criterion)
        
        # 评估另一个模型
        print("\n评估模型2...")
        other_evaluator = Evaluator(other_model, self.device)
        metrics2 = other_evaluator.evaluate(dataloader, criterion)
        
        # 比较
        comparison = {
            'model1': metrics1,
            'model2': metrics2,
            'accuracy_diff': metrics1['accuracy'] - metrics2['accuracy'],
            'f1_diff': metrics1['f1'] - metrics2['f1'],
            'loss_diff': metrics1['loss'] - metrics2['loss']
        }
        
        print("\n=== 模型对比 ===")
        print(f"准确率差异: {comparison['accuracy_diff']:.4f}")
        print(f"F1分数差异: {comparison['f1_diff']:.4f}")
        print(f"损失差异: {comparison['loss_diff']:.4f}")
        
        return comparison
