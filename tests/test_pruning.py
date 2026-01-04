"""
剪枝模块测试
"""

import unittest
import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerModel
from src.pruning.structured_pruning import StructuredPruning, AttentionHeadPruner, FFNPruner
from src.pruning.unstructured_pruning import UnstructuredPruning, MagnitudePruner
from src.pruning.progressive_pruning import ProgressivePruning


class TestStructuredPruning(unittest.TestCase):
    """结构化剪枝测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            num_classes=2
        )
        self.device = torch.device('cpu')
        
    def test_attention_head_pruner_init(self):
        """测试注意力头剪枝器初始化"""
        pruner = AttentionHeadPruner(self.model)
        self.assertIsNotNone(pruner)
        
    def test_ffn_pruner_init(self):
        """测试FFN剪枝器初始化"""
        pruner = FFNPruner(self.model)
        self.assertIsNotNone(pruner)
        
    def test_structured_pruning_init(self):
        """测试结构化剪枝初始化"""
        pruner = StructuredPruning(self.model)
        self.assertIsNotNone(pruner)


class TestUnstructuredPruning(unittest.TestCase):
    """非结构化剪枝测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            num_classes=2
        )
        
    def test_magnitude_pruner_init(self):
        """测试幅度剪枝器初始化"""
        pruner = MagnitudePruner(self.model)
        self.assertIsNotNone(pruner)
        
    def test_get_prunable_parameters(self):
        """测试获取可剪枝参数"""
        pruner = MagnitudePruner(self.model)
        params = pruner.get_prunable_parameters()
        self.assertGreater(len(params), 0)
        
    def test_compute_weight_importance(self):
        """测试计算权重重要性"""
        pruner = MagnitudePruner(self.model)
        weight = torch.randn(10, 10)
        importance = pruner.compute_weight_importance(weight, 'magnitude')
        self.assertEqual(importance.shape, weight.shape)
        
    def test_unstructured_pruning_init(self):
        """测试非结构化剪枝初始化"""
        pruner = UnstructuredPruning(self.model)
        self.assertIsNotNone(pruner)
        
    def test_apply_pruning(self):
        """测试应用剪枝"""
        pruner = UnstructuredPruning(self.model)
        
        # 记录原始参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # 应用剪枝
        stats = pruner.apply_pruning(sparsity=0.3, global_pruning=True)
        
        # 验证统计信息
        self.assertIn('total_params', stats)
        self.assertIn('pruned_params', stats)
        self.assertIn('actual_sparsity', stats)
        
        # 验证稀疏度
        sparsity = pruner.get_sparsity()
        self.assertGreater(sparsity, 0)
        self.assertLess(sparsity, 1)


class TestProgressivePruning(unittest.TestCase):
    """渐进式剪枝测试"""
    
    def setUp(self):
        """测试前准备"""
        self.model = TransformerModel(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            num_classes=2
        )
        self.config = {
            'initial_sparsity': 0.0,
            'final_sparsity': 0.5,
            'num_iterations': 5
        }
        
    def test_progressive_pruning_init(self):
        """测试渐进式剪枝初始化"""
        pruner = ProgressivePruning(self.model, self.config)
        self.assertIsNotNone(pruner)
        self.assertEqual(pruner.initial_sparsity, 0.0)
        self.assertEqual(pruner.final_sparsity, 0.5)
        
    def test_compute_target_sparsity(self):
        """测试计算目标稀疏度"""
        pruner = ProgressivePruning(self.model, self.config)
        
        # 第一次迭代
        sparsity_0 = pruner.compute_target_sparsity(0)
        self.assertGreaterEqual(sparsity_0, self.config['initial_sparsity'])
        
        # 最后一次迭代
        sparsity_final = pruner.compute_target_sparsity(self.config['num_iterations'])
        self.assertAlmostEqual(sparsity_final, self.config['final_sparsity'])
        
    def test_get_pruning_schedule(self):
        """测试获取剪枝计划"""
        pruner = ProgressivePruning(self.model, self.config)
        schedule = pruner.get_pruning_schedule()
        
        self.assertEqual(len(schedule), self.config['num_iterations'])
        self.assertIsInstance(schedule, dict)


if __name__ == '__main__':
    unittest.main()
