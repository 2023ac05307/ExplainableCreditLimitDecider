from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch

from .model_loader import ModelBundle
from .inference import _vectorize, _scale


@dataclass
class ExplainConfig:
    top_k: int = 8
    baseline: str = "zeros"  # zeros or mean
    steps: int = 32          # for integrated gradients fallback


def _topk(feature_names: List[str], values: np.ndarray, attrs: np.ndarray, k: int):
    idx = np.argsort(np.abs(attrs))[::-1][:k]
    out = []
    for i in idx:
        out.append((feature_names[i], float(values[i]), float(attrs[i])))
    return out


def _integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 32,
) -> torch.Tensor:
    """
    Simple IG for single-output models (logit).
    Returns attributions with same shape as x.
    """
    assert x.shape == baseline.shape
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device).view(-1, 1)
    x_interp = baseline + alphas * (x - baseline)
    x_interp.requires_grad_(True)

    # forward
    out = model(x_interp)
    y = _as_logit(out)

    # make it scalar per sample
    if y.ndim > 1:
        y = y[:, 0]

    y_sum = y.sum()

    # backward
    grads = torch.autograd.grad(y_sum, x_interp, retain_graph=False, create_graph=False)[0]  # (steps, d)

    avg_grads = grads.mean(dim=0, keepdim=True)  # (1, d)
    attrs = (x - baseline) * avg_grads
    return attrs



def _as_logit(out: object) -> torch.Tensor:
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        for k in ["logit", "logits", "pi_logit", "pi_logits", "policy_logit", "policy_logits", "actor_logit", "actor_logits"]:
            if k in out and torch.is_tensor(out[k]):
                return out[k]
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




def explain_one(
    bundle: ModelBundle,
    features: Dict[str, float],
    stage: str = "final",
    top_k: int = 8,
) -> Tuple[str, List[Tuple[str, float, float]], str, Dict[str, Any]]:
    """
    Returns:
      method, [(feature, value, attribution)], explanation_text, meta
    """
    # Choose which model to explain
    if stage == "gate":
        mdl = bundle.gate
        model_name = "gate"
    elif stage == "dir":
        mdl = bundle.dir
        model_name = "dir"
    else:
        # for "final", explain the gate decision by default (most interpretable)
        mdl = bundle.gate
        model_name = "gate"

    cols = mdl.state_cols
    x_np = _vectorize(features, cols)
    x_np_scaled = _scale(x_np, mdl.scaler_mean, mdl.scaler_std)

    x = torch.from_numpy(x_np_scaled).unsqueeze(0).to(mdl.device)

    # Baseline
    base_np = np.zeros_like(x_np_scaled, dtype=np.float32)
    
    baseline = torch.from_numpy(base_np).unsqueeze(0).to(mdl.device)

    # Try SHAP if present
    try:
        import shap  # type: ignore

        # Use a tiny background set: baseline + x (works, but not perfect; good enough for demo)
        background = torch.cat([baseline, x], dim=0)
        def f(x):
            out = mdl.model(x)
            logit = _as_logit(out)
            # ensure shape [B,1] or [B] consistently
            return logit.view(logit.shape[0], -1)

        explainer = shap.GradientExplainer(f, background)

        shap_vals = explainer.shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        attrs = np.array(shap_vals).reshape(-1).astype(np.float32)
        method = "shap_gradient"
    except Exception:
        # Fallback to IG
        mdl.model.zero_grad(set_to_none=True)
        attrs_t = _integrated_gradients(mdl.model, x, baseline, steps=32)
        attrs = attrs_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        method = "integrated_gradients"

    # Map attributions back to unscaled feature values (more readable)
    top = _topk(cols, x_np, attrs, top_k)

    # Simple explanation text
    lines = []
    for feat, val, a in top:
        direction = "increased" if a > 0 else "decreased"
        lines.append(f"{feat} ({val:.4g}) {direction} the {model_name} score (impact={a:.4g}).")

    meta = {"model": model_name, "stage": stage}
    explanation = "\n".join(lines)
    return method, top, explanation, meta
