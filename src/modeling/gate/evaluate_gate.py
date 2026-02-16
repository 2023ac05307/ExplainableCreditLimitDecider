#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCriticBinary(nn.Module):
    """Must match gate trainer architecture."""
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_logit = nn.Linear(hidden, 1)
        # q/v may or may not exist in ckpt; we load with strict=False when needed
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        return logit, q, v

def load_gate_ckpt(ckpt_path: str, device: str = DEVICE) -> Tuple[nn.Module, list[str], np.ndarray, np.ndarray, float]:
    """
    Loads Gate checkpoint.

    Supports:
      - Option A: ckpt["model"] is nn.Module
      - Legacy: ckpt["model"] is state_dict
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_obj = ck.get("model", None)
    if isinstance(model_obj, nn.Module):
        model = model_obj.to(device)
        model.eval()
    else:
        # Legacy fallback: require rebuilding model (only if you still have old ckpts)
        # If you don't need legacy, you can delete this block.
        from src.modeling.gate.model import GateActorCritic, GateModelConfig
        obs_dim = int(ck["obs_dim"])
        cfg = GateModelConfig(obs_dim=obs_dim)
        model = GateActorCritic(cfg, include_q=True, include_v=True).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

    state_cols = ck["state_cols"]
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32)
    thr = float(ck.get("best_thr", 0.5))
    return model, state_cols, mu, sd, thr




def load_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)


def _safe_load_state(model: nn.Module, sdict: dict) -> None:
    try:
        model.load_state_dict(sdict, strict=True)
    except Exception:
        model.load_state_dict(sdict, strict=False)

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # binary: 0=HOLD, 1=NONHOLD
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, eps)
    tpr = rec
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (tpr + tnr)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_acc": float(bal_acc),
        **{k: float(v) for k, v in c.items()},
    }


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, mu: np.ndarray, sd: np.ndarray, batch_size: int) -> np.ndarray:
    mu_t = torch.as_tensor(mu, device=DEVICE)
    sd_t = torch.as_tensor(sd, device=DEVICE)
    out = np.zeros(X.shape[0], dtype=np.float32)

    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t
        logit, _, _ = model(xz)
        out[i:i+batch_size] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="gated_val.parquet (or gated_test.parquet)")
    ap.add_argument("--ckpt", required=True, help="Gate checkpoint .pt")
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--label-col", default="y_gate", help="Binary label col: 0=HOLD, 1=NONHOLD")
    ap.add_argument("--out-json", default="reports/eval/gate_metrics.json")
    ap.add_argument("--out-parquet", default="", help="Optional parquet with probabilities/preds")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    df = load_parquet_any(args.data)
    if args.label_col not in df.columns:
        raise RuntimeError(f"Missing label col {args.label_col} in {args.data}")

    model, cols, mu, sd = load_gate_ckpt(args.ckpt)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required state cols (first 20): {missing[:20]}")

    X = df[cols].astype(np.float32).to_numpy(copy=True)
    y_true = df[args.label_col].astype(int).to_numpy()
    p_nonhold = predict_proba(model, X, mu, sd, args.batch_size)
    y_pred = (p_nonhold >= args.thr).astype(int)

    m = metrics_binary(y_true, y_pred)
    out = {
        "task": "gate",
        "threshold": args.thr,
        "rows": int(len(df)),
        "metrics": m,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if args.out_parquet:
        odf = df.copy()
        odf["p_nonhold"] = p_nonhold
        odf["pred_nonhold"] = y_pred
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        odf.to_parquet(args.out_parquet, index=False)
        print(f"Saved: {args.out_parquet}")


if __name__ == "__main__":
    main()
