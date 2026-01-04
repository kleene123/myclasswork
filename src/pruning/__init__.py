"""
剪枝模块
"""

from .structured_pruning import StructuredPruning, AttentionHeadPruner, FFNPruner
from .unstructured_pruning import UnstructuredPruning, MagnitudePruner
from .progressive_pruning import ProgressivePruning

__all__ = [
    "StructuredPruning",
    "AttentionHeadPruner",
    "FFNPruner",
    "UnstructuredPruning",
    "MagnitudePruner",
    "ProgressivePruning",
]
