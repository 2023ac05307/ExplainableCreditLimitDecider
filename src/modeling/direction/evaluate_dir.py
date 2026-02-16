#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_dir.py
---------------
Evaluate Direction model (CLD vs CLI) on TRUE NONHOLD dataset.

This version aligns metrics + reporting style with evaluate_gate.py:
  - adds confusion counts helper
  - adds balanced accuracy (useful for class imbalance)
  - supports using checkpoint's best_thr when --thr is not provided

Checkpoint compatibility:
  - Option A: ckpt["model"] is nn.Module
  - Legacy: ckpt["model"] is a state_dict
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.modeling.direction.model import DirActorCritic, DirModelConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)


def load_dir_ckpt(ckpt_path: str) -> Tuple[nn.Module, list[str], np.ndarray, np.ndarray, Optional[float]]:
    """Load DIR checkpoint and return (model, state_cols, scaler_mean, scaler_std, best_thr_or_none)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    obs_dim = int(ck["obs_dim"])
    state_cols = ck["state_cols"]
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32)
    best_thr = ck.get("best_thr", None)
    best_thr = float(best_thr) if best_thr is not None else None

    m = ck.get("model", None)

    # Option A: full module stored
    if isinstance(m, nn.Module):
        model = m.to(DEVICE)
        model.eval()
        return model, state_cols, mu, sd, best_thr

    # Legacy: state_dict stored in ck["model"]
    cfg = DirModelConfig(obs_dim=obs_dim)
    model = DirActorCritic(cfg, include_q=True, include_v=True).to(DEVICE)
    try:
        model.load_state_dict(ck["model"], strict=True)
    except Exception:
        model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, state_cols, mu, sd, best_thr


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # binary: 0=CLD, 1=CLI
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return gate-aligned metrics dict."""
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    eps = 1e-12
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, eps)

    # balanced accuracy = (TPR + TNR) / 2
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
def predict_proba_cli(
    model: nn.Module,
    X: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    batch_size: int,
    invert_prob: bool = False,
) -> np.ndarray:
    mu_t = torch.as_tensor(mu, device=DEVICE, dtype=torch.float32)
    sd_t = torch.as_tensor(sd, device=DEVICE, dtype=torch.float32)

    out = np.zeros(X.shape[0], dtype=np.float32)

    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t

        logits = model(xz)["logit"]
        p = torch.sigmoid(logits)

        # ✅ FIX: if model output is oriented as P(CLD), convert to P(CLI)
        if invert_prob:
            p = 1.0 - p

        out[i:i + batch_size] = p.detach().cpu().numpy().astype(np.float32)

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir_val.parquet (true NONHOLD only)")
    ap.add_argument("--ckpt", required=True, help="DIR checkpoint .pt")

    ap.add_argument(
        "--thr",
        type=float,
        default=None,
        help="Threshold for predicting CLI (1). If omitted, uses ckpt['best_thr'] when available, else 0.5.",
    )

    ap.add_argument("--label-col", default="y_dir", help="Binary label col: CLD=0, CLI=1")
    ap.add_argument("--out-json", default="reports/eval/dir_metrics.json")
    ap.add_argument("--out-parquet", default="", help="Optional parquet with probabilities/preds")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--invert-prob",action="store_true",help="If set, treats sigmoid(logit) as P(CLD) and uses p_cli = 1 - sigmoid(logit).")

    args = ap.parse_args()

    df = load_parquet_any(args.data)
    if args.label_col not in df.columns:
        raise RuntimeError(f"Missing label col {args.label_col} in {args.data}")

    model, cols, mu, sd, ckpt_thr = load_dir_ckpt(args.ckpt)
    thr = float(args.thr) if args.thr is not None else (ckpt_thr if ckpt_thr is not None else 0.5)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required state cols (first 20): {missing[:20]}")

    X = df[cols].astype(np.float32).to_numpy(copy=True)
    y_true = df[args.label_col].astype(int).to_numpy()

    p_cli = predict_proba_cli(model, X, mu, sd, args.batch_size, invert_prob=args.invert_prob)

    y_pred = (p_cli >= thr).astype(int)

    m = metrics_binary(y_true, y_pred)
    out = {"task": "dir", "threshold": float(thr), "rows": int(len(df)), "metrics": m}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if args.out_parquet:
        odf = df.copy()
        odf["p_cli"] = p_cli
        odf["pred_cli"] = y_pred
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        odf.to_parquet(args.out_parquet, index=False)
        print(f"Saved: {args.out_parquet}")


if __name__ == "__main__":
    main()
