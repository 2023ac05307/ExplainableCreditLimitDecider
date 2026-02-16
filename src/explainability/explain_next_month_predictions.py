#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_next_month_action_limit.py
----------------------------------
Two-step inference for next-month credit action + split magnitude models:

  1) Gate model: HOLD(0) vs NON_HOLD(1)
  2) Dir model : CLD(0) vs CLI(1) only if NON_HOLD
  3) Magnitude:
        - if final action == CLI -> mag_cli_ckpt
        - if final action == CLD -> mag_cld_ckpt

Inputs:
  - CSV with snapshot features (s_*) for next month
Outputs:
  - CSV with predicted next month action + magnitude

NOTE about torch.load:
  - PyTorch 2.6+ changed torch.load default weights_only to True
  - Your ckpts contain numpy arrays => require weights_only=False
  - Use only for trusted ckpts (your own training outputs).

Example:
python predict_next_month_action_limit.py \
  --in_csv  data/next_month_snapshot.csv \
  --out_csv out/next_month_decisions.csv \
  --gate_ckpt checkpoints/classification/two_step_gate_checkpoint_subsample.pt \
  --dir_ckpt  checkpoints/classification/two_step_dir_checkpoint_subsample.pt \
  --mag_cli_ckpt checkpoints/regression/mag_beta_cli_regression.pt \
  --mag_cld_ckpt checkpoints/regression/mag_beta_cld_regression.pt \
  --gate_thr 0.65 --dir_thr 0.22 \
  --mag_scale_pct 40 \
  --write_probs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import uuid



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3-class convention
A_HOLD, A_CLI, A_CLD = 0, 1, 2


# =============================================================================
# Utilities
# =============================================================================

def _torch_load_trusted(path: Path) -> Dict:
    """Trusted loader for ckpts that may include numpy objects."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


def ensure_cols(df: pd.DataFrame, cols: List[str], ctx: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ctx} Missing required columns: {missing[:25]}{'...' if len(missing) > 25 else ''}")


def to_numeric_f32(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Convert selected columns to float32 safely and fill NaNs with 0."""
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32).to_numpy(copy=True)


def normalize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.clip(sd, 1e-6, None)
    return (X - mu[None, :]) / sd[None, :]


def load_any_parquet(path: str) -> pd.DataFrame:
    """
    Supports:
      - single .parquet file
      - parquet dataset directory (part-*.parquet)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if p.is_dir():
        return pd.read_parquet(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)

    raise RuntimeError(f"Unsupported input format (expected parquet): {path}")


def save_any_parquet(df: pd.DataFrame, out_path: str) -> str:
    """
    Safe writer:
      - If out_path ends with .parquet -> single file
      - Else -> directory with part-*.parquet
    """
    p = Path(out_path)
    if p.suffix.lower() == ".parquet":
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        return str(p)
    else:
        p.mkdir(parents=True, exist_ok=True)
        part = p / f"part-{uuid.uuid4().hex}.parquet"
        df.to_parquet(part, index=False)
        return str(p)


# =============================================================================
# Models (must match your trainers)
# =============================================================================

class ActorCriticBinary(nn.Module):
    """Gate binary model architecture (matches your 2-step trainers)."""
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_logit = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        return logit, q, v


class ActorCriticDirBinaryNoMag(nn.Module):
    """DIR model without alpha/beta heads."""
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_logit = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        return logit, q, v


class ActorCriticDirBinaryWithMag(nn.Module):
    """DIR model with alpha/beta heads (optional; we don't rely on it when split mag ckpts exist)."""
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi_logit = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        self.mag_alpha = nn.Linear(hidden, 1)
        self.mag_beta = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        alpha = F.softplus(self.mag_alpha(x)).squeeze(-1) + 1.0
        beta = F.softplus(self.mag_beta(x)).squeeze(-1) + 1.0
        return logit, q, v, alpha, beta


class BetaRegressor(nn.Module):
    """
    Standalone magnitude model (matches your train_mag_beta_regression_parquet.py):
      forward -> (mu, phi)
      mu in (0,1)
    """
    def __init__(self, obs_dim: int, hidden: int = 256, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(d, 1)
        self.phi_head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        phi = F.softplus(self.phi_head(h)).squeeze(-1) + 1e-3
        return mu, phi


# =============================================================================
# Loaders
# =============================================================================

def load_gate_awac(ckpt_path: Path, hidden: int, dropout: float) -> Tuple[nn.Module, List[str], np.ndarray, np.ndarray]:
    ck = _torch_load_trusted(ckpt_path)
    state_cols = list(ck["state_cols"])
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32).reshape(-1)
    obs_dim = int(ck.get("obs_dim", len(state_cols)))

    model = ActorCriticBinary(obs_dim=obs_dim, hidden=hidden, dropout=dropout).to(DEVICE)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, state_cols, mu, sd


def load_dir_awac(ckpt_path: Path, hidden: int, dropout: float) -> Tuple[nn.Module, bool, List[str], np.ndarray, np.ndarray]:
    ck = _torch_load_trusted(ckpt_path)
    state_cols = list(ck["state_cols"])
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32).reshape(-1)
    obs_dim = int(ck.get("obs_dim", len(state_cols)))

    sdict = ck["model"]
    has_mag = any(k.startswith("mag_alpha") or k.startswith("mag_beta") for k in sdict.keys())

    if has_mag:
        model = ActorCriticDirBinaryWithMag(obs_dim=obs_dim, hidden=hidden, dropout=dropout).to(DEVICE)
        model.load_state_dict(sdict, strict=True)
    else:
        model = ActorCriticDirBinaryNoMag(obs_dim=obs_dim, hidden=hidden, dropout=dropout).to(DEVICE)
        model.load_state_dict(sdict, strict=True)

    model.eval()
    return model, has_mag, state_cols, mu, sd


def load_mag_beta(ckpt_path: Path) -> Tuple[nn.Module, List[str], np.ndarray, np.ndarray, Dict]:
    """
    Loads beta regression ckpt produced by your train_mag_beta_regression_parquet.py:
      - model_state
      - feature_cols
      - scaler_mean/std
      - config: hidden, depth, dropout, etc.
    """
    ck = _torch_load_trusted(ckpt_path)

    feat_cols = list(ck["feature_cols"])
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32).reshape(-1)

    cfg = ck.get("config", {})
    hidden = int(cfg.get("hidden", 256))
    depth = int(cfg.get("depth", 3))
    dropout = float(cfg.get("dropout", 0.0))

    model = BetaRegressor(obs_dim=len(feat_cols), hidden=hidden, depth=depth, dropout=dropout).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    return model, feat_cols, mu, sd, cfg


# =============================================================================
# Batched inference
# =============================================================================

@torch.no_grad()
def run_gate(model: nn.Module, Xn: np.ndarray, batch_size: int) -> np.ndarray:
    p_nonhold = np.zeros(len(Xn), dtype=np.float32)
    for i in range(0, len(Xn), batch_size):
        xb = torch.from_numpy(Xn[i:i + batch_size]).to(DEVICE)
        logit, _, _ = model(xb)
        p = torch.sigmoid(logit)
        p_nonhold[i:i + batch_size] = p.detach().cpu().numpy().astype(np.float32)
    return p_nonhold


@torch.no_grad()
def run_dir(model: nn.Module, has_mag_heads: bool, Xn: np.ndarray, batch_size: int) -> np.ndarray:
    p_cli = np.zeros(len(Xn), dtype=np.float32)
    for i in range(0, len(Xn), batch_size):
        xb = torch.from_numpy(Xn[i:i + batch_size]).to(DEVICE)
        if has_mag_heads:
            logit, _, _, _, _ = model(xb)  # type: ignore[misc]
        else:
            logit, _, _ = model(xb)
        p = torch.sigmoid(logit)
        p_cli[i:i + batch_size] = p.detach().cpu().numpy().astype(np.float32)
    return p_cli


@torch.no_grad()
def run_mag_mu01(model: nn.Module, Xn: np.ndarray, batch_size: int) -> np.ndarray:
    mu01 = np.zeros(len(Xn), dtype=np.float32)
    for i in range(0, len(Xn), batch_size):
        xb = torch.from_numpy(Xn[i:i + batch_size]).to(DEVICE)
        m, _ = model(xb)  # (mu, phi)
        mu01[i:i + batch_size] = m.detach().cpu().numpy().astype(np.float32)
    return mu01


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_parquet", type=str, required=True,
                    help="Input parquet file OR parquet dataset directory")
    ap.add_argument("--out_parquet", type=str, required=True,
                    help="Output parquet file OR directory")


    ap.add_argument("--gate_ckpt", type=str, required=True)
    ap.add_argument("--dir_ckpt", type=str, required=True)

    # Split magnitude models
    ap.add_argument("--mag_cli_ckpt", type=str, required=True, help="Beta regression ckpt for CLI (action=1)")
    ap.add_argument("--mag_cld_ckpt", type=str, required=True, help="Beta regression ckpt for CLD (action=2)")

    ap.add_argument("--feature_prefix", type=str, default="s_")

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)

    ap.add_argument("--gate_thr", type=float, default=0.50)
    ap.add_argument("--dir_thr", type=float, default=0.50)

    ap.add_argument("--batch_size", type=int, default=4096)

    # How to map mu01 -> pct_points:
    # If your magnitude targets are fraction of 40% (0..0.40), then mu01 corresponds to (0..1) over 40%.
    ap.add_argument("--mag_scale_pct", type=float, default=40.0,
                    help="Scale to convert mu01 (0..1) into % points. Use 40 if targets are 0..0.40.")

    ap.add_argument("--mag_out", choices=["frac", "pct"], default="frac",
                    help="Backward-compat output in next_month_magnitude_pct column.")

    ap.add_argument("--write_probs", action="store_true")

    args = ap.parse_args()

    df = load_any_parquet(args.in_parquet)

    if len(df) == 0:
        raise RuntimeError("Input parquet is empty.")

    # --------------------
    # Load models
    # --------------------
    gate_m, gate_cols, gmu, gsd = load_gate_awac(Path(args.gate_ckpt), hidden=args.hidden, dropout=args.dropout)
    dir_m, dir_has_mag, dir_cols, dmu, dsd = load_dir_awac(Path(args.dir_ckpt), hidden=args.hidden, dropout=args.dropout)

    mag_cli_m, mag_cli_cols, mag_cli_mu, mag_cli_sd, _ = load_mag_beta(Path(args.mag_cli_ckpt))
    mag_cld_m, mag_cld_cols, mag_cld_mu, mag_cld_sd, _ = load_mag_beta(Path(args.mag_cld_ckpt))

    # --------------------
    # Prepare features
    # --------------------
    ensure_cols(df, gate_cols, ctx="Gate:")
    Xg = normalize(to_numeric_f32(df, gate_cols), gmu, gsd)

    ensure_cols(df, dir_cols, ctx="Dir:")
    Xd = normalize(to_numeric_f32(df, dir_cols), dmu, dsd)

    ensure_cols(df, mag_cli_cols, ctx="Mag(CLI):")
    X_cli_mag = normalize(to_numeric_f32(df, mag_cli_cols), mag_cli_mu, mag_cli_sd)

    ensure_cols(df, mag_cld_cols, ctx="Mag(CLD):")
    X_cld_mag = normalize(to_numeric_f32(df, mag_cld_cols), mag_cld_mu, mag_cld_sd)

    # --------------------
    # Gate -> Dir -> Final action
    # --------------------
    p_nonhold = run_gate(gate_m, Xg, batch_size=args.batch_size)
    pred_nonhold = (p_nonhold >= float(args.gate_thr)).astype(np.int32)

    p_cli = np.zeros(len(df), dtype=np.float32)
    nonhold_idx = np.where(pred_nonhold == 1)[0]
    if len(nonhold_idx) > 0:
        p_cli[nonhold_idx] = run_dir(dir_m, dir_has_mag, Xd[nonhold_idx], batch_size=args.batch_size)

    pred_cli = (p_cli >= float(args.dir_thr)).astype(np.int32)

    final_action = np.full(len(df), A_HOLD, dtype=np.int32)
    final_action[nonhold_idx] = np.where(pred_cli[nonhold_idx] == 1, A_CLI, A_CLD)

    # --------------------
    # Split magnitude routing
    # --------------------
    final_mag_pct_points = np.zeros(len(df), dtype=np.float32)
    final_mag_frac = np.zeros(len(df), dtype=np.float32)

    cli_mask = (final_action == A_CLI)
    cld_mask = (final_action == A_CLD)

    if np.any(cli_mask):
        mu01_cli = run_mag_mu01(mag_cli_m, X_cli_mag[cli_mask], batch_size=args.batch_size)
        cli_pct = np.clip(mu01_cli * float(args.mag_scale_pct), 0.0, float(args.mag_scale_pct))
        final_mag_pct_points[cli_mask] = cli_pct.astype(np.float32)
        final_mag_frac[cli_mask] = (cli_pct / 100.0).astype(np.float32)

    if np.any(cld_mask):
        mu01_cld = run_mag_mu01(mag_cld_m, X_cld_mag[cld_mask], batch_size=args.batch_size)
        cld_pct = np.clip(mu01_cld * float(args.mag_scale_pct), 0.0, float(args.mag_scale_pct))
        final_mag_pct_points[cld_mask] = cld_pct.astype(np.float32)
        final_mag_frac[cld_mask] = (cld_pct / 100.0).astype(np.float32)

    # --------------------
    # Write output
    # --------------------
    out = df.copy()
    out["next_month_action_3cls"] = final_action
    out["next_month_action_name"] = np.where(
        final_action == A_HOLD, "HOLD", np.where(final_action == A_CLI, "CLI", "CLD")
    )

    out["next_month_magnitude_frac"] = final_mag_frac
    out["next_month_magnitude_pct_points"] = final_mag_pct_points
    out["next_month_magnitude_pct"] = final_mag_frac if args.mag_out == "frac" else final_mag_pct_points

    if args.write_probs:
        out["gate_prob_nonhold"] = p_nonhold.astype(np.float32)
        out["gate_prob_hold"] = (1.0 - p_nonhold).astype(np.float32)
        out["dir_prob_cli"] = p_cli.astype(np.float32)
        out["dir_prob_cld"] = (1.0 - p_cli).astype(np.float32)
        out["gate_thr_used"] = float(args.gate_thr)
        out["dir_thr_used"] = float(args.dir_thr)

    saved_to = save_any_parquet(out, args.out_parquet)
    print(f"[OK] Wrote parquet to: {saved_to}")


    counts = pd.Series(final_action).value_counts().reindex([0, 1, 2], fill_value=0).to_dict()
    counts = pd.Series(final_action).value_counts().reindex([0, 1, 2], fill_value=0).to_dict()
    print(f"Pred action counts: {counts} (0=HOLD,1=CLI,2=CLD)")
    print(f"Gate thr={args.gate_thr:.3f} | Dir thr={args.dir_thr:.3f} | mag_scale_pct={args.mag_scale_pct:.1f}")

if __name__ == "__main__":
    main()
