from __future__ import annotations

from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field, ConfigDict


ActionTaken = Literal["HOLD", "CLI", "CLD"]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cust_id: int = Field(..., ge=0)
    # Features as a name->value map (server will reorder using state_cols from ckpt)
    features: Dict[str, float]



class ExplainAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str                 # <-- MUST be str
    value: float
    attribution: float


class ExplainCustomerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cust_id: str
    action_taken: ActionTaken
    method: str
    explanation_lines: List[str]
    attributions: List[ExplainAttribution]
    meta: Dict[str, Any]


class ExplainCustomerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[ExplainCustomerItem]



class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cust_id: int
    next_month: str
    action_taken: ActionTaken
    magnitude_percentage: float
    prev_credit_limit: float
    updated_credit_limit: float
    # Useful for debugging/analysis (can hide in production if needed)
    gate_prob: float
    dir_prob: Optional[float] = None


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[PredictResponse]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cust_id: int = Field(..., ge=0)
    features: Dict[str, float]
    top_k: int = Field(8, ge=1, le=30)
    # Explain which stage
    stage: Literal["gate", "dir", "final"] = "final"


class FeatureAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    value: float
    attribution: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cust_id: int
    action_taken: ActionTaken
    explanation_lines: List[str]
    attributions: List[FeatureAttribution]
    method: str
    meta: Dict[str, Any] = Field(default_factory=dict)
