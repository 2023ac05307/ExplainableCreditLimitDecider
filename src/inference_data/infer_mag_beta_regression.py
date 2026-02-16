#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_mag_beta_regression.py
----------------------------
Inference + evaluation for magnitude Beta regression.

- Uses original TEST data only
- Filters HOLD rows
- Predicts continuous magnitude in %
- Computes MAE in percentage points
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Model definition (must match training)
# -----------------------------

class BetaRegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        self.phi_head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.backbone(x)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        phi = F.softplus(self.phi_head(h)).squeeze(-1) + 1e-3
        return mu, phi


# -----------------------------
# Main
# -----------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # ---- Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")

    feature_cols = ckpt["feature_cols"]
    scaler_mean = np.array(ckpt["scaler_mean"])
    scaler_std = np.array(ckpt["scaler_std"])
    cfg = ckpt["config"]

    # ---- Load test data
    df = pd.read_csv(args.test)

    # ---- Filter NONHOLD only
    df = df[df[args.action_col].astype(int) != args.hold_id].copy()

    # ---- Check features
    DROP_IF_MISSING = {"sample_weight", "w", "weight"}

    feature_cols_infer = [c for c in feature_cols if c in df.columns and c not in DROP_IF_MISSING]
    dropped = [c for c in feature_cols if c not in df.columns or c in DROP_IF_MISSING]

    if len(feature_cols_infer) == 0:
        raise ValueError("No usable feature columns found for inference after dropping missing/training-only cols.")

    if dropped:
        print(f"⚠️ Dropping {len(dropped)} feature(s) not present/allowed in test: {dropped[:10]}")
    feature_cols = feature_cols_infer

    # ---- Prepare X
    X = df[feature_cols].astype(float).to_numpy()
    X = (X - scaler_mean) / scaler_std
    X = torch.from_numpy(X).float().to(device)

    # ---- True magnitude (for evaluation)
    y_true_pct = np.abs(df[args.target_col].astype(float).to_numpy())

    # ---- Load model
    model = BetaRegressor(
        d_in=X.shape[1],
        hidden=cfg["hidden"],
        depth=cfg["depth"]
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Inference
    with torch.no_grad():
        mu, _ = model(X)
        pred_frac = mu.cpu().numpy()
        pred_pct = pred_frac * 100.0

    # ---- Evaluation
    y_true_frac = np.abs(df[args.target_col].astype(float).to_numpy())
    y_true_pct = y_true_frac * 100.0
    mae = np.mean(np.abs(pred_pct - y_true_pct))

    print("\n===== MAGNITUDE MODEL TEST EVALUATION =====")
    print(f"Test rows (NONHOLD): {len(df)}")
    print(f"MAE (percentage points): {mae:.4f}")
    print(f"Predicted % min / max / mean: "
          f"{pred_pct.min():.2f} / {pred_pct.max():.2f} / {pred_pct.mean():.2f}")

    # ---- Save predictions
    out_df = df.copy()
    out_df["pred_magnitude_pct"] = pred_pct
    out_df["abs_error_pct"] = np.abs(pred_pct - y_true_pct)

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved predictions to: {args.out}")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Trained magnitude checkpoint (.pt)")
    ap.add_argument("--test", required=True, help="Original TEST CSV")
    ap.add_argument("--out", required=True, help="Output CSV with predictions")
    ap.add_argument("--action-col", default="action_id")
    ap.add_argument("--hold-id", type=int, default=0)
    ap.add_argument("--target-col", default="action_delta_pct")
    ap.add_argument("--cpu", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
