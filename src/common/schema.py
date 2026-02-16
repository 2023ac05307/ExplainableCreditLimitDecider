from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class DataSchemaError(ValueError):
    """Raised when incoming data violates expected schema."""


@dataclass
class PredictionRow:
    """
    One customer prediction for next month.
    """
    cust_id: int
    next_month: str

    action_taken: str  # HOLD / CLI / CLD
    magnitude_percentage: float

    prev_credit_limit: float
    updated_credit_limit: float

    explanation: str = ""


@dataclass
class PredictionBatch:
    """
    Prediction response for a batch.
    """
    model_version: str
    rows: List[PredictionRow] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "rows": [r.__dict__ for r in self.rows],
            "extra": self.extra,
        }


@dataclass
class ModelMeta:
    """
    Metadata for a model artifact.
    """
    name: str
    task: str
    version: str
    trained_at: str
    framework: str = "pytorch"
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointMeta:
    """
    Unified metadata for checkpoint bundles.
    Helps report + debugging.
    """
    task: str
    ckpt_format: str  # "option_a" / "state_dict" / "beta_regression"
    obs_dim: Optional[int] = None
    state_cols: Optional[List[str]] = None
    scaler_mean: Optional[Any] = None
    scaler_std: Optional[Any] = None
    best_epoch: Optional[int] = None
    best_thr: Optional[float] = None
    best_val_f1: Optional[float] = None
    notes: str = ""
