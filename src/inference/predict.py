#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-step inference (PARQUET):
  1) Gate model: HOLD(0) vs NON_HOLD(1)
  2) Dir model : CLD(0) vs CLI(1) only if NON_HOLD

Optional:
  - Write gated_test.parquet and dir_test.parquet from 3-class ground truth
  - Use separate magnitude model checkpoint (Beta regression) to predict magnitude for CLI/CLD

INPUT  : Parquet file OR Parquet dataset directory
OUTPUT : Parquet file OR Parquet dataset directory (safe writer)

Labels:
  0=HOLD, 1=CLI, 2=CLD
"""

import argparse
import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

A_HOLD, A_CLI, A_CLD = 0, 1, 2
MAG_MAX_DEFAULT = 0.40  # used only by dir-head magnitude fallback


# -----------------------------
# Gate + Dir architectures (must match trainers)
# -----------------------------

class ActorCriticBinary(nn.Module):
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
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        return logit, q, v


class ActorCriticDirBinary(nn.Module):
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


class ActorCriticDirBinaryNoMag(nn.Module):
    """DIR model without alpha/beta magnitude heads (has_mag=False checkpoints)."""
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
        self.q = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logit = self.pi_logit(x).squeeze(-1)
        q = self.q(x)
        v = self.v(x).squeeze(-1)
        return logit, q, v


# -----------------------------
# Magnitude model (Beta regression)
# -----------------------------

class BetaRegressor(nn.Module):
    """
    Magnitude model checkpoint trained separately.
    Outputs mu in (0,1) and phi>0; we use mu as the predicted mean.
    """
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
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)
        phi = F.softplus(self.phi_head(h)).squeeze(-1) + 1e-3
        return mu, phi


# -----------------------------
# Parquet I/O
# -----------------------------

def load_parquet_any(path: str) -> pd.DataFrame:
    """
    Supports:
      - single .parquet file
      - parquet dataset directory (part-*.parquet)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    df = pd.read_parquet(path)
    return df.replace([np.inf, -np.inf], np.nan)


def save_parquet_any(df: pd.DataFrame, out_path: str):
    """
    Safe writer:
      - If out_path endswith .parquet -> writes a single parquet file
      - Else treats out_path as a directory and writes a part-*.parquet inside it
    Avoids WinError 5 when a directory path is passed.
    """
    outp = Path(out_path)
    if outp.suffix.lower() == ".parquet":
        outp.parent.mkdir(parents=True, exist_ok=True)
        tmp = outp.with_suffix(outp.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, outp)
        return str(outp)
    else:
        outp.mkdir(parents=True, exist_ok=True)
        part = outp / f"part-{uuid.uuid4().hex}.parquet"
        df.to_parquet(part, index=False)
        return str(outp)


# -----------------------------
# Loaders
# -----------------------------

def load_gate(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)

    model = ActorCriticBinary(obs_dim).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, state_cols, mu, sd


def load_dir(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)
    mag_max = float(ck.get("mag_max", MAG_MAX_DEFAULT))

    sdict = ck["model"]
    # NOTE: original code had a small bug; keeping your intent but fixing the check safely
    has_mag = any(k.startswith(("mag_alpha", "mag_beta")) for k in sdict.keys())

    if has_mag:
        model = ActorCriticDirBinary(obs_dim).to(DEVICE)
        model.load_state_dict(sdict, strict=True)
    else:
        model = ActorCriticDirBinaryNoMag(obs_dim).to(DEVICE)
        model.load_state_dict(sdict, strict=True)

    model.eval()
    return model, state_cols, mu, sd, mag_max, has_mag


def load_mag(ckpt_path: str):
    """
    Expected keys:
      - model_state
      - feature_cols
      - scaler_mean, scaler_std
      - config {hidden, depth, max_pct}
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_cols = ck["feature_cols"]
    mu = np.array(ck["scaler_mean"], dtype=np.float32)
    sd = np.array(ck["scaler_std"], dtype=np.float32)
    cfg = ck.get("config", {})
    hidden = int(cfg.get("hidden", 256))
    depth = int(cfg.get("depth", 3))
    max_pct = float(cfg.get("max_pct", 50.0))

    model = BetaRegressor(obs_dim=len(feat_cols), hidden=hidden, depth=depth).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, feat_cols, mu, sd, max_pct


def _safe_load_state(model: nn.Module, sdict: dict, name: str):
    try:
        model.load_state_dict(sdict, strict=True)
        return
    except RuntimeError as e:
        # Backward-compat: some ckpts don't store q/v heads; we only need pi_logit for inference.
        missing, unexpected = model.load_state_dict(sdict, strict=False)
        print(f"⚠️ [{name}] Loaded with strict=False for inference compatibility.")
        if missing:
            print(f"   Missing keys (initialized randomly, NOT used for pi inference): {missing}")
        if unexpected:
            print(f"   Unexpected keys (ignored): {unexpected}")

def load_gate(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)

    model = ActorCriticBinary(obs_dim).to(DEVICE)
    _safe_load_state(model, ck["model"], name="GATE")
    model.eval()
    return model, state_cols, mu, sd

def load_dir(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)
    mag_max = float(ck.get("mag_max", MAG_MAX_DEFAULT))

    sdict = ck["model"]
    has_mag = any(k.startswith(("mag_alpha", "mag_beta")) for k in sdict.keys())

    if has_mag:
        model = ActorCriticDirBinary(obs_dim).to(DEVICE)
        _safe_load_state(model, sdict, name="DIR(has_mag)")
    else:
        model = ActorCriticDirBinaryNoMag(obs_dim).to(DEVICE)
        _safe_load_state(model, sdict, name="DIR(no_mag)")

    model.eval()
    return model, state_cols, mu, sd, mag_max, has_mag


# -----------------------------
# Helpers
# -----------------------------

def ensure_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns: {missing[:20]}{'...' if len(missing)>20 else ''}")


def build_gate_and_dir_tests(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise RuntimeError(f"Label column '{label_col}' not found in input parquet.")

    y3 = df[label_col].astype(int)

    bad = y3[~y3.isin([A_HOLD, A_CLI, A_CLD])]
    if len(bad) > 0:
        ex = bad.iloc[:10].tolist()
        raise RuntimeError(f"Unexpected label values in '{label_col}'. Expected only 0/1/2. Examples: {ex}")

    gated = df.copy()
    gated["y_gate"] = (y3 != A_HOLD).astype(np.int32)
    gated["y_gate_name"] = np.where(gated["y_gate"].to_numpy() == 1, "NON_HOLD", "HOLD")

    nh_mask = (y3 != A_HOLD).to_numpy()
    dir_df = df.loc[nh_mask].copy()
    y3_nh = y3.loc[nh_mask].to_numpy()
    dir_df["y_dir"] = (y3_nh == A_CLI).astype(np.int32)   # CLI=1, CLD=0
    dir_df["y_dir_name"] = np.where(dir_df["y_dir"].to_numpy() == 1, "CLI", "CLD")

    return gated, dir_df


# -----------------------------
# Batch predictors
# -----------------------------

@torch.no_grad()
def batch_predict_gate(model, X: np.ndarray, mu: np.ndarray, sd: np.ndarray, batch_size: int):
    p_nonhold = np.zeros(X.shape[0], dtype=np.float32)
    mu_t = torch.as_tensor(mu, device=DEVICE)
    sd_t = torch.as_tensor(sd, device=DEVICE)
    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t
        logit, _, _ = model(xz)
        p_nonhold[i:i+batch_size] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)
    return p_nonhold


@torch.no_grad()
def batch_predict_dir(model, X: np.ndarray, mu: np.ndarray, sd: np.ndarray, mag_max: float, batch_size: int, has_mag: bool):
    p_cli = np.zeros(X.shape[0], dtype=np.float32)
    mag_mean = np.zeros(X.shape[0], dtype=np.float32)  # fallback only if has_mag

    mu_t = torch.as_tensor(mu, device=DEVICE)
    sd_t = torch.as_tensor(sd, device=DEVICE)

    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t

        if has_mag:
            logit, _, _, alpha, beta = model(xz)
            mag_mean[i:i+batch_size] = (alpha/(alpha+beta)).detach().cpu().numpy().astype(np.float32) * float(mag_max)
        else:
            logit, _, _ = model(xz)

        p_cli[i:i+batch_size] = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)

    return p_cli, mag_mean


@torch.no_grad()
def batch_predict_mag(model, X: np.ndarray, mu: np.ndarray, sd: np.ndarray, batch_size: int):
    out_mu = np.zeros(X.shape[0], dtype=np.float32)
    mu_t = torch.as_tensor(mu, device=DEVICE)
    sd_t = torch.as_tensor(sd, device=DEVICE)
    for i in range(0, X.shape[0], batch_size):
        xb = torch.as_tensor(X[i:i+batch_size], device=DEVICE, dtype=torch.float32)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        xz = (xb - mu_t) / sd_t
        mu_pred, _ = model(xz)
        out_mu[i:i+batch_size] = mu_pred.detach().cpu().numpy().astype(np.float32)
    return out_mu


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="Parquet file OR parquet dataset directory")
    ap.add_argument("--gate_ckpt", required=True)
    ap.add_argument("--dir_ckpt", required=True)
    ap.add_argument("--out_parquet", required=True, help="Parquet file OR output directory")

    ap.add_argument("--gate_thr", type=float, default=0.50)
    ap.add_argument("--dir_thr", type=float, default=0.50)
    ap.add_argument("--batch_size", type=int, default=8192)

    ap.add_argument("--label_col", default="final_action_3cls")
    ap.add_argument("--gated_test_parquet", default="")
    ap.add_argument("--dir_test_parquet", default="")

    # Magnitude checkpoint (optional)
    # Magnitude checkpoints (optional, split)
    ap.add_argument("--mag_cli_ckpt", default="", help="Magnitude model for CLI action (1)")
    ap.add_argument("--mag_cld_ckpt", default="", help="Magnitude model for CLD action (2)")

    ap.add_argument("--mag_mode", choices=["frac", "pct"], default="frac",
                    help="frac: model mu is fraction")
    ap.add_argument("--mag_out", choices=["frac", "pct"], default="frac",
                    help="What to store in final_magnitude_pct (backward compat).")
    args = ap.parse_args()

    df = load_parquet_any(args.in_parquet)

    # Optional derived tests
    if args.gated_test_parquet or args.dir_test_parquet:
        gated_df, dir_df = build_gate_and_dir_tests(df, args.label_col)
        if args.gated_test_parquet:
            save_parquet_any(gated_df, args.gated_test_parquet)
            print("✅ Saved gated test:", args.gated_test_parquet)
        if args.dir_test_parquet:
            save_parquet_any(dir_df, args.dir_test_parquet)
            print("✅ Saved dir test:", args.dir_test_parquet)

    gate_model, gate_cols, gate_mu, gate_sd = load_gate(args.gate_ckpt)
    dir_model, dir_cols, dir_mu, dir_sd, dir_mag_max, dir_has_mag = load_dir(args.dir_ckpt)

    ensure_cols(df, gate_cols)
    ensure_cols(df, dir_cols)

    X_gate = df[gate_cols].fillna(0.0).astype(np.float32).to_numpy(copy=True)
    X_dir  = df[dir_cols].fillna(0.0).astype(np.float32).to_numpy(copy=True)

    # Optional mag model
    mag_cli = mag_cld = None
    mag_cli_cols = mag_cli_mu = mag_cli_sd = None
    mag_cld_cols = mag_cld_mu = mag_cld_sd = None
    mag_cli_max_pct = mag_cld_max_pct = None

    use_split_mag = bool(args.mag_cli_ckpt) and bool(args.mag_cld_ckpt)

    if use_split_mag:
        mag_cli, mag_cli_cols, mag_cli_mu, mag_cli_sd, mag_cli_max_pct = load_mag(args.mag_cli_ckpt)
        mag_cld, mag_cld_cols, mag_cld_mu, mag_cld_sd, mag_cld_max_pct = load_mag(args.mag_cld_ckpt)

        ensure_cols(df, mag_cli_cols)
        ensure_cols(df, mag_cld_cols)

    elif args.mag_cli_ckpt or args.mag_cld_ckpt:
        raise RuntimeError("Provide BOTH --mag_cli_ckpt and --mag_cld_ckpt (or none).")

    # 1) Gate
    p_nonhold = batch_predict_gate(gate_model, X_gate, gate_mu, gate_sd, args.batch_size)
    pred_nonhold = (p_nonhold >= args.gate_thr).astype(np.int32)

    # 2) Dir
    p_cli, mag_mean_dir = batch_predict_dir(dir_model, X_dir, dir_mu, dir_sd, dir_mag_max, args.batch_size, dir_has_mag)
    pred_cli = (p_cli >= args.dir_thr).astype(np.int32)

    final_action = np.full(len(df), A_HOLD, dtype=np.int32)
    nh_mask = pred_nonhold == 1
    final_action[nh_mask] = np.where(pred_cli[nh_mask] == 1, A_CLI, A_CLD)

    
    # Magnitude outputs
    final_mag_frac = np.zeros(len(df), dtype=np.float32)
    final_mag_pct_points = np.zeros(len(df), dtype=np.float32)

    cli_mask = final_action == A_CLI
    cld_mask = final_action == A_CLD

    if use_split_mag:
        # --- CLI magnitude ---
        if np.any(cli_mask):
            X_mag_cli = df[mag_cli_cols].fillna(0.0).astype(np.float32).to_numpy(copy=True)
            mu_cli = batch_predict_mag(mag_cli, X_mag_cli, mag_cli_mu, mag_cli_sd, args.batch_size)  # (0,1)

            # trained as y = pct_points/40  =>  pct_points = mu * 40
            cli_pct_points = np.clip(mu_cli * 40.0, 0.0, 40.0)

            final_mag_pct_points[cli_mask] = cli_pct_points[cli_mask].astype(np.float32)
            final_mag_frac[cli_mask] = (cli_pct_points[cli_mask] / 100.0).astype(np.float32)

        # --- CLD magnitude ---
        if np.any(cld_mask):
            X_mag_cld = df[mag_cld_cols].fillna(0.0).astype(np.float32).to_numpy(copy=True)
            mu_cld = batch_predict_mag(mag_cld, X_mag_cld, mag_cld_mu, mag_cld_sd, args.batch_size)  # (0,1)

            # trained as y = pct_points/40  =>  pct_points = mu * 40
            cld_pct_points = np.clip(mu_cld * 40.0, 0.0, 40.0)

            final_mag_pct_points[cld_mask] = cld_pct_points[cld_mask].astype(np.float32)
            final_mag_frac[cld_mask] = (cld_pct_points[cld_mask] / 100.0).astype(np.float32)

        mag_info = "MAG=split(cli,cld) scale=40"

    else:
        # fallback to dir-head magnitude (existing behavior)
        final_mag_frac[nh_mask] = mag_mean_dir[nh_mask].astype(np.float32)
        final_mag_pct_points[nh_mask] = (mag_mean_dir[nh_mask] * 100.0).astype(np.float32)
        mag_info = f"MAG=dir_head mag_max={dir_mag_max:.2f}"



    out = df.copy()
    out["p_nonhold"] = p_nonhold
    out["p_hold"] = (1.0 - p_nonhold).astype(np.float32)
    out["p_cli"] = p_cli
    out["p_cld"] = (1.0 - p_cli).astype(np.float32)
    out["pred_nonhold"] = pred_nonhold
    out["pred_cli"] = pred_cli
    out["final_action_3cls"] = final_action

    # Backward compatible column name + explicit ones
    out["final_magnitude_frac"] = final_mag_frac
    out["final_magnitude_pct_points"] = final_mag_pct_points
    out["final_magnitude_pct"] = final_mag_frac if args.mag_out == "frac" else final_mag_pct_points

    saved_to = save_parquet_any(out, args.out_parquet)

    print("Saved preds:", saved_to)
    print(f"Gate thr={args.gate_thr:.2f} | Dir thr={args.dir_thr:.2f} | {mag_info}")
    print("Pred final action distribution:",
          out["final_action_3cls"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0).to_dict())


if __name__ == "__main__":
    main()
