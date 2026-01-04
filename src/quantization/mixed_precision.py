"""
混合精度量化实现
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import copy


class MixedPrecisionQuantization:
    """混合精度量化类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化混合精度量化
        
        Args:
            model: 要量化的模型
            config: 量化配置
        """
        self.model = model
        self.config = config or {}
        
        self.default_dtype = self.config.get('default_dtype', 'int8')
        self.sensitive_dtype = self.config.get('sensitive_dtype', 'fp16')
        self.sensitive_layers = self.config.get('sensitive_layers', [])
        
        self.layer_sensitivity = {}
        
    def analyze_layer_sensitivity(
        self,
        dataloader,
        criterion: nn.Module,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        分析各层对量化的敏感度
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_samples: 用于分析的样本数
            
        Returns:
            层敏感度字典 {layer_name: sensitivity_score}
        """
        print("分析层敏感度...")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # 获取基线损失
        baseline_loss = self._compute_loss(dataloader, criterion, num_samples, device)
        print(f"基线损失: {baseline_loss:.4f}")
        
        # 遍历每一层，单独量化并测试
        layer_sensitivity = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 保存原始权重
                original_weight = module.weight.data.clone()
                
                # 模拟量化（简化版：使用FP16）
                module.weight.data = module.weight.data.half().float()
                
                # 计算量化后的损失
                quantized_loss = self._compute_loss(dataloader, criterion, num_samples, device)
                
                # 计算敏感度（损失增加量）
                sensitivity = quantized_loss - baseline_loss
                layer_sensitivity[name] = sensitivity
                
                # 恢复原始权重
                module.weight.data = original_weight
                
                print(f"层 {name}: 敏感度 = {sensitivity:.4f}")
        
        self.layer_sensitivity = layer_sensitivity
        
        return layer_sensitivity
    
    def _compute_loss(
        self,
        dataloader,
        criterion: nn.Module,
        num_samples: int,
        device: torch.device
    ) -> float:
        """
        计算模型在数据集上的平均损失
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            num_samples: 样本数
            device: 设备
            
        Returns:
            平均损失
        """
        total_loss = 0.0
        samples_processed = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                labels = batch['labels'].to(device)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
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
                
                # 计算损失
                loss = criterion(logits, labels)
                total_loss += loss.item() * input_ids.size(0)
                samples_processed += input_ids.size(0)
        
        avg_loss = total_loss / samples_processed if samples_processed > 0 else 0
        return avg_loss
    
    def select_sensitive_layers(
        self,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        选择最敏感的层
        
        Args:
            top_k: 选择前k个最敏感的层
            threshold: 敏感度阈值（可选）
            
        Returns:
            敏感层名称列表
        """
        if not self.layer_sensitivity:
            raise ValueError("请先运行 analyze_layer_sensitivity()")
        
        # 按敏感度排序
        sorted_layers = sorted(
            self.layer_sensitivity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择top-k或超过阈值的层
        if threshold is not None:
            sensitive_layers = [name for name, sens in sorted_layers if sens > threshold]
        else:
            sensitive_layers = [name for name, _ in sorted_layers[:top_k]]
        
        self.sensitive_layers = sensitive_layers
        
        print(f"\n选择的敏感层 ({len(sensitive_layers)}):")
        for layer in sensitive_layers:
            print(f"  - {layer}: {self.layer_sensitivity[layer]:.4f}")
        
        return sensitive_layers
    
    def apply_mixed_precision(
        self,
        sensitive_layers: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        应用混合精度量化
        
        Args:
            sensitive_layers: 敏感层列表（使用高精度）
            
        Returns:
            层配置字典 {layer_name: dtype}
        """
        if sensitive_layers is None:
            sensitive_layers = self.sensitive_layers
        
        print("\n应用混合精度量化...")
        
        layer_config = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if name in sensitive_layers:
                    # 敏感层使用高精度
                    if self.sensitive_dtype == 'fp16':
                        module.half()
                    layer_config[name] = self.sensitive_dtype
                    print(f"层 {name}: {self.sensitive_dtype} (敏感层)")
                else:
                    # 非敏感层使用低精度
                    # 注意：实际的INT8量化需要更复杂的处理
                    layer_config[name] = self.default_dtype
                    print(f"层 {name}: {self.default_dtype}")
        
        return layer_config
    
    def get_precision_distribution(self) -> Dict[str, int]:
        """
        获取精度分布统计
        
        Returns:
            精度分布字典 {dtype: count}
        """
        distribution = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if name in self.sensitive_layers:
                    dtype = self.sensitive_dtype
                else:
                    dtype = self.default_dtype
                
                distribution[dtype] = distribution.get(dtype, 0) + 1
        
        print("\n=== 混合精度分布 ===")
        for dtype, count in distribution.items():
            print(f"{dtype}: {count} 层")
        
        return distribution
    
    def estimate_model_size(self) -> Dict[str, float]:
        """
        估算混合精度模型的大小
        
        Returns:
            大小统计字典
        """
        total_params = 0
        int8_params = 0
        fp16_params = 0
        fp32_params = 0
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            # 根据配置确定精度
            layer_name = '.'.join(name.split('.')[:-1])  # 移除 'weight' 或 'bias'
            
            if layer_name in self.sensitive_layers:
                if self.sensitive_dtype == 'fp16':
                    fp16_params += num_params
                else:
                    fp32_params += num_params
            else:
                if self.default_dtype == 'int8':
                    int8_params += num_params
                elif self.default_dtype == 'fp16':
                    fp16_params += num_params
                else:
                    fp32_params += num_params
        
        # 计算大小（字节）
        size_int8 = int8_params * 1  # 1 byte per param
        size_fp16 = fp16_params * 2  # 2 bytes per param
        size_fp32 = fp32_params * 4  # 4 bytes per param
        
        total_size_mb = (size_int8 + size_fp16 + size_fp32) / 1024 / 1024
        
        stats = {
            'total_params': total_params,
            'int8_params': int8_params,
            'fp16_params': fp16_params,
            'fp32_params': fp32_params,
            'total_size_mb': total_size_mb,
            'int8_size_mb': size_int8 / 1024 / 1024,
            'fp16_size_mb': size_fp16 / 1024 / 1024,
            'fp32_size_mb': size_fp32 / 1024 / 1024
        }
        
        print(f"\n=== 混合精度模型大小估算 ===")
        print(f"总参数: {total_params:,}")
        print(f"INT8 参数: {int8_params:,} ({size_int8/1024/1024:.2f} MB)")
        print(f"FP16 参数: {fp16_params:,} ({size_fp16/1024/1024:.2f} MB)")
        print(f"FP32 参数: {fp32_params:,} ({size_fp32/1024/1024:.2f} MB)")
        print(f"总大小: {total_size_mb:.2f} MB")
        
        return stats
