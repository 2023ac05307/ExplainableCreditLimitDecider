"""
Datasets shared across modeling scripts (AWAC/PPO/IQL).

This package exists so training scripts can import:
    from src.modeling.data.datasets import TrajDatasetGATE, TrajDatasetDIR
"""

from .datasets import TrajDatasetGATE, TrajDatasetDIR, infer_columns

__all__ = ["TrajDatasetGATE", "TrajDatasetDIR", "infer_columns"]
