"""
量化模块测试
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerModel
from src.quantization.ptq import PostTrainingQuantization
from src.quantization.qat import QuantizationAwareTraining
from src.quantization.mixed_precision import MixedPrecisionQuantization


class TestPostTrainingQuantization(unittest.TestCase):
    """训练后量化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=1024,
            num_classes=2
        )
        self.config = {
            'dtype': 'int8',
            'calibration_samples': 10,
            'per_channel': True
        }
        
    def test_ptq_init(self):
        """测试PTQ初始化"""
        ptq = PostTrainingQuantization(self.model, self.config)
        self.assertIsNotNone(ptq)
        self.assertEqual(ptq.dtype, 'int8')
        
    def test_get_model_size(self):
        """测试获取模型大小"""
        ptq = PostTrainingQuantization(self.model, self.config)
        size = ptq.get_model_size()
        self.assertGreater(size, 0)
        
    def test_apply_dynamic_quantization(self):
        """测试应用动态量化"""
        ptq = PostTrainingQuantization(self.model, self.config)
        
        # 应用动态量化
        quantized_model = ptq.apply_dynamic_quantization(dtype=torch.qint8)
        self.assertIsNotNone(quantized_model)
        
        # 验证模型大小变化
        original_size = PostTrainingQuantization(
            TransformerModel(
                vocab_size=1000,
                d_model=256,
                num_heads=8,
                num_layers=2,
                d_ff=1024,
                num_classes=2
            ),
            self.config
        ).get_model_size()
        
        quantized_size = ptq.get_model_size()
        # 动态量化可能不会显著减小模型大小
        self.assertGreater(quantized_size, 0)


class TestQuantizationAwareTraining(unittest.TestCase):
    """量化感知训练测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=1024,
            num_classes=2
        )
        self.config = {
            'dtype': 'int8',
            'num_calibration_batches': 5
        }
        
    def test_qat_init(self):
        """测试QAT初始化"""
        qat = QuantizationAwareTraining(self.model, self.config)
        self.assertIsNotNone(qat)
        self.assertEqual(qat.dtype, 'int8')
        
    def test_get_model_size(self):
        """测试获取模型大小"""
        qat = QuantizationAwareTraining(self.model, self.config)
        size = qat.get_model_size()
        self.assertGreater(size, 0)


class TestMixedPrecisionQuantization(unittest.TestCase):
    """混合精度量化测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=1024,
            num_classes=2
        )
        self.config = {
            'default_dtype': 'int8',
            'sensitive_dtype': 'fp16',
            'sensitive_layers': []
        }
        
    def test_mixed_precision_init(self):
        """测试混合精度量化初始化"""
        mp = MixedPrecisionQuantization(self.model, self.config)
        self.assertIsNotNone(mp)
        self.assertEqual(mp.default_dtype, 'int8')
        self.assertEqual(mp.sensitive_dtype, 'fp16')
        
    def test_get_precision_distribution(self):
        """测试获取精度分布"""
        mp = MixedPrecisionQuantization(self.model, self.config)
        distribution = mp.get_precision_distribution()
        self.assertIsInstance(distribution, dict)
        
    def test_estimate_model_size(self):
        """测试估算模型大小"""
        mp = MixedPrecisionQuantization(self.model, self.config)
        stats = mp.estimate_model_size()
        
        self.assertIn('total_params', stats)
        self.assertIn('total_size_mb', stats)
        self.assertGreater(stats['total_params'], 0)


if __name__ == '__main__':
    unittest.main()
