"""
训练后量化 (Post-Training Quantization) 实现
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Optional, List
from torch.utils.data import DataLoader


class PostTrainingQuantization:
    """训练后量化类"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        初始化训练后量化
        
        Args:
            model: 要量化的模型
            config: 量化配置
        """
        self.model = model
        self.config = config or {}
        
        self.dtype = self.config.get('dtype', 'int8')
        self.calibration_samples = self.config.get('calibration_samples', 1000)
        self.per_channel = self.config.get('per_channel', True)
        self.reduce_range = self.config.get('reduce_range', False)
        
    def prepare_model_for_quantization(self):
        """
        准备模型进行量化
        
        设置量化配置并插入观察器
        """
        # 设置为评估模式
        self.model.eval()
        
        # 配置量化
        if self.dtype == 'int8':
            if self.per_channel:
                # 每通道量化配置
                self.model.qconfig = quant.get_default_qconfig('fbgemm')
            else:
                # 每张量量化配置
                self.model.qconfig = quant.default_qconfig
        else:
            # FP16 使用不同的配置
            self.model.qconfig = quant.float16_static_qconfig
        
        # 准备量化（插入观察器）
        if hasattr(self.model, 'model'):
            # BERTWrapper
            quant.prepare(self.model.model, inplace=True)
        else:
            quant.prepare(self.model, inplace=True)
        
        print("模型已准备好进行量化")
    
    def calibrate(self, dataloader: DataLoader, num_samples: int = None):
        """
        使用校准数据集收集统计信息
        
        Args:
            dataloader: 校准数据加载器
            num_samples: 用于校准的样本数量
        """
        if num_samples is None:
            num_samples = self.calibration_samples
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        print(f"开始校准，使用 {num_samples} 个样本...")
        
        samples_processed = 0
        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break
                
                # 准备输入
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # 前向传播以收集统计信息
                if hasattr(self.model, 'model'):
                    _ = self.model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                else:
                    _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                samples_processed += input_ids.size(0)
        
        print(f"校准完成，处理了 {samples_processed} 个样本")
    
    def convert_to_quantized(self):
        """
        将模型转换为量化模型
        
        Returns:
            量化后的模型
        """
        print("转换模型为量化格式...")
        
        if hasattr(self.model, 'model'):
            # BERTWrapper
            quant.convert(self.model.model, inplace=True)
        else:
            quant.convert(self.model, inplace=True)
        
        print("模型已转换为量化格式")
        
        return self.model
    
    def apply_dynamic_quantization(
        self,
        dtype: torch.dtype = torch.qint8
    ):
        """
        应用动态量化
        
        动态量化只量化权重，激活在运行时动态量化
        
        Args:
            dtype: 量化数据类型
            
        Returns:
            量化后的模型
        """
        print("应用动态量化...")
        
        # 指定要量化的层类型
        layers_to_quantize = [nn.Linear]
        
        if hasattr(self.model, 'model'):
            quantized_model = quant.quantize_dynamic(
                self.model.model,
                layers_to_quantize,
                dtype=dtype
            )
            self.model.model = quantized_model
        else:
            quantized_model = quant.quantize_dynamic(
                self.model,
                layers_to_quantize,
                dtype=dtype
            )
            self.model = quantized_model
        
        print("动态量化完成")
        
        return self.model
    
    def apply_static_quantization(
        self,
        calibration_dataloader: DataLoader
    ):
        """
        应用静态量化
        
        静态量化需要校准数据集来确定激活的量化参数
        
        Args:
            calibration_dataloader: 校准数据加载器
            
        Returns:
            量化后的模型
        """
        print("应用静态量化...")
        
        # 准备模型
        self.prepare_model_for_quantization()
        
        # 校准
        self.calibrate(calibration_dataloader, self.calibration_samples)
        
        # 转换
        quantized_model = self.convert_to_quantized()
        
        print("静态量化完成")
        
        return quantized_model
    
    def get_model_size(self) -> float:
        """
        获取量化后的模型大小（MB）
        
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
    
    def compare_with_fp32(self, fp32_model) -> Dict[str, any]:
        """
        与FP32模型比较
        
        Args:
            fp32_model: FP32精度的原始模型
            
        Returns:
            比较统计信息
        """
        # 获取量化模型大小
        quant_size = self.get_model_size()
        
        # 获取FP32模型大小
        fp32_size = 0
        for param in fp32_model.parameters():
            fp32_size += param.nelement() * param.element_size()
        for buffer in fp32_model.buffers():
            fp32_size += buffer.nelement() * buffer.element_size()
        fp32_size = fp32_size / 1024 / 1024
        
        compression_ratio = fp32_size / quant_size if quant_size > 0 else 0
        
        stats = {
            'fp32_size_mb': fp32_size,
            'quantized_size_mb': quant_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - quant_size / fp32_size) * 100 if fp32_size > 0 else 0
        }
        
        print(f"\n=== 量化对比 ===")
        print(f"FP32 模型大小: {fp32_size:.2f} MB")
        print(f"量化模型大小: {quant_size:.2f} MB")
        print(f"压缩比: {compression_ratio:.2f}x")
        print(f"大小减少: {stats['size_reduction_percent']:.2f}%")
        
        return stats
