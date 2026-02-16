from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import time
from .model_loader import ModelBundle
from .metrics import PREDICTIONS,INFERENCE_STAGE_LATENCY


@dataclass
class InferenceConfig:
    gate_threshold: float = 0.5
    dir_threshold: float = 0.5
    # magnitude output assumptions:
    # - if model outputs fraction (0..1), you can convert to percent outside
    mag_output_is_percent: bool = True
    # clamp magnitude to [0, max_pct]
    max_pct: float = 40.0


def _vectorize(features: Dict[str, float], state_cols: list[str]) -> np.ndarray:
    x = np.zeros((len(state_cols),), dtype=np.float32)
    for i, c in enumerate(state_cols):
        v = features.get(c, 0.0)
        try:
            x[i] = float(v)
        except Exception:
            x[i] = 0.0
    return x


def _scale(x: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return x
    std_safe = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std_safe

def _as_logit(out):
    """
    Accepts model output as:
      - Tensor: return as-is
      - dict: return first matching key
      - tuple/list: return first tensor element
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        # try common keys (adjust if your gate uses different names)
        for k in ["logit", "logits", "pi_logit", "pi_logits", "policy_logit", "policy_logits", "actor_logit", "actor_logits"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        # fallback: first tensor value in dict
        for v in out.values():
            if torch.is_tensor(v):
                return v
        raise TypeError(f"Dict output has no Tensor values. Keys={list(out.keys())[:20]}")

    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
        raise TypeError("Tuple/list output has no Tensor element.")

    raise TypeError(f"Unsupported model output type: {type(out)}")

@torch.no_grad()
def predict_one(
    bundle: ModelBundle,
    features: Dict[str, float],
    prev_credit_limit: float,
    next_month: str,
    cfg: InferenceConfig = InferenceConfig(),
) -> Tuple[str, float, float, Optional[float], float]:
    """
    Returns:
      action_taken, magnitude_pct, updated_limit, dir_prob, gate_prob
    """

    t0 = time.perf_counter()
    # Gate
    g = bundle.gate
    xg = _vectorize(features, g.state_cols)
    xg = _scale(xg, g.scaler_mean, g.scaler_std)
    xt = torch.from_numpy(xg).unsqueeze(0).to(g.device)

    gate_out = g.model(xt)
    gate_logit = _as_logit(gate_out).view(-1)[0]
    gate_prob = torch.sigmoid(gate_logit).item()

    if gate_prob < cfg.gate_threshold:
        action = "HOLD"
        mag_pct = 0.0
        updated = float(prev_credit_limit)
        PREDICTIONS.labels(action=action).inc()
        return action, mag_pct, updated, None, float(gate_prob)
    
    INFERENCE_STAGE_LATENCY.labels(stage="gate").observe(time.perf_counter() - t0)

    t1 = time.perf_counter()

    # Dir (only if NONHOLD)
    d = bundle.dir
    xd = _vectorize(features, d.state_cols)
    xd = _scale(xd, d.scaler_mean, d.scaler_std)
    xtd = torch.from_numpy(xd).unsqueeze(0).to(d.device)

    dir_out = d.model(xtd)
    dir_logit = _as_logit(dir_out).view(-1)[0]
    dir_prob = torch.sigmoid(dir_logit).item()
    action = "CLI" if dir_prob >= cfg.dir_threshold else "CLD"
    INFERENCE_STAGE_LATENCY.labels(stage="dir").observe(time.perf_counter() - t1)

    # Magnitude
    if action == "CLI":
        m = bundle.mag_cli
    else:
        m = bundle.mag_cld

    t2 = time.perf_counter()
    # ---- Magnitude model (BetaRegressor) ----
    m = bundle.mag_cli if action == "CLI" else bundle.mag_cld

    xm = _vectorize(features, m.feature_cols).astype("float32")

    # apply saved standardization (same as training)
    xm = (xm - m.scaler_mean) / (m.scaler_std + 1e-8)

    xtm = torch.from_numpy(xm).unsqueeze(0).to(m.device)


    mu, phi = m.model(xtm)  # mu in (0,1)

    # max_pct must be in percentage points (e.g., 40.0)
    max_pct = float(
        (m.config.get("max_pct", 40.0) if hasattr(m, "config") and isinstance(m.config, dict) else 40.0)
    )

    # IMPORTANT normalization: if someone stored 0.40 meaning 40%, fix it
    if 0 < max_pct <= 1.0:
        max_pct *= 100.0

    mag_pct = float(mu.item()) * max_pct      # percentage points (0..40)
    mag_frac = mag_pct / 100.0                # fraction (0..0.40)

    if action == "CLI":
        updated = prev_credit_limit * (1.0 + mag_frac)
    else:
        updated = prev_credit_limit * (1.0 - mag_frac)
    INFERENCE_STAGE_LATENCY.labels(stage="beta").observe(time.perf_counter() - t2)

    PREDICTIONS.labels(action=action).inc()
    return action, mag_pct, float(updated), float(dir_prob), float(gate_prob)
