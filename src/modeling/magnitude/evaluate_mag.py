from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BetaRegressor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        self.phi_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)       # (0,1)
        phi = F.softplus(self.phi_head(h)).squeeze(-1) + 1e-3 # >0
        return mu, phi


def load_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path).replace([np.inf, -np.inf], np.nan)


def load_mag_ckpt(ckpt_path: str) -> Tuple[nn.Module, list[str], np.ndarray, np.ndarray, float]:
    """
    Expected keys:
      - model_state
      - feature_cols
      - scaler_mean, scaler_std
      - config {hidden, depth, max_pct} (max_pct in percentage points, e.g., 40 or 50)
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_cols = ck["feature_cols"]
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32)
    cfg = ck.get("config", {})
    hidden = int(cfg.get("hidden", 256))
    depth = int(cfg.get("depth", 3))
    max_pct = float(cfg.get("max_pct", 40.0))

    model = BetaRegressor(obs_dim=len(feat_cols), hidden=hidden, depth=depth).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, feat_cols, mu, sd, max_pct


@torch.no_grad()
def predict_mu(model: nn.Module, X: np.ndarray, mu: np.ndarray, sd: np.ndarray, batch_size: int) -> np.ndarray:
    mu_t = torch.as_tensor(mu, device=DEVICE)
    sd_t = torch.as_tensor(sd, device=DEVICE)
    out = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t
        mu_pred, _ = model(xz)
        out[i:i+batch_size] = mu_pred.detach().cpu().numpy().astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="trajectories_val/test parquet containing true actions + target")
    ap.add_argument("--ckpt", required=True, help="Magnitude checkpoint .pt")
    ap.add_argument("--action-col", default="action_id")
    ap.add_argument("--action-id", type=int, required=True, help="Evaluate only this action: 1=CLI, 2=CLD")
    ap.add_argument("--target-col", default="action_delta_pct",
                    help="Target should be FRACTION (e.g. 0.12 for 12pp). If stored as pp, adjust below.")
    ap.add_argument("--target-is-percent-points", action="store_true",
                    help="Set if target-col already stores % points (e.g. 12.0 meaning 12pp).")
    ap.add_argument("--out-json", default="reports/eval/mag_metrics.json")
    ap.add_argument("--out-parquet", default="")
    ap.add_argument("--batch-size", type=int, default=8192)
    args = ap.parse_args()

    df = load_parquet_any(args.data)
    if args.action_col not in df.columns:
        raise RuntimeError(f"Missing action col {args.action_col}")
    if args.target_col not in df.columns:
        raise RuntimeError(f"Missing target col {args.target_col}")

    df = df[df[args.action_col].astype(int) == int(args.action_id)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for action_id={args.action_id} in {args.data}")

    model, feat_cols, s_mu, s_sd, max_pct = load_mag_ckpt(args.ckpt)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature cols (first 20): {missing[:20]}")

    X = df[feat_cols].astype(np.float32).to_numpy(copy=True)

    # truth in % points
    y = df[args.target_col].astype(float).to_numpy()
    if args.target_is_percent_points:
        y_true_pp = np.abs(y)
    else:
        # fraction -> % points
        y_true_pp = np.abs(y) * 100.0

    mu_pred = predict_mu(model, X, s_mu, s_sd, args.batch_size)  # (0,1)

    # convert to predicted % points using checkpoint max_pct
    y_pred_pp = np.clip(mu_pred * max_pct, 0.0, max_pct)

    mae = float(np.mean(np.abs(y_pred_pp - y_true_pp)))
    rmse = float(np.sqrt(np.mean((y_pred_pp - y_true_pp) ** 2)))

    out = {
        "task": "magnitude",
        "action_id": int(args.action_id),
        "rows": int(len(df)),
        "max_pct_from_ckpt": float(max_pct),
        "mae_pp": mae,
        "rmse_pp": rmse,
        "pred_pp_min": float(np.min(y_pred_pp)),
        "pred_pp_max": float(np.max(y_pred_pp)),
        "pred_pp_mean": float(np.mean(y_pred_pp)),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if args.out_parquet:
        odf = df.copy()
        odf["pred_mag_mu"] = mu_pred
        odf["pred_mag_pp"] = y_pred_pp
        odf["abs_error_pp"] = np.abs(y_pred_pp - y_true_pp)
        Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
        odf.to_parquet(args.out_parquet, index=False)
        print(f"Saved: {args.out_parquet}")


if __name__ == "__main__":
    main()
