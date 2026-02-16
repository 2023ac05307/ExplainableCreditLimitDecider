# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Device & speed flags
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# Model
# -----------------------------
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = torch.sigmoid(self.mu_head(h)).squeeze(-1)           # (0,1)
        phi = F.softplus(self.phi_head(h)).squeeze(-1) + 1e-3     # >0
        return mu, phi


def beta_nll(y01: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    y01 = torch.clamp(y01, 1e-6, 1.0 - 1e-6)
    alpha = torch.clamp(mu * phi, 1e-6, 1e9)
    beta = torch.clamp((1.0 - mu) * phi, 1e-6, 1e9)

    logB = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    ll = (alpha - 1.0) * torch.log(y01) + (beta - 1.0) * torch.log(1.0 - y01) - logB
    return -ll


# -----------------------------
# Dataset
# -----------------------------
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Config
# -----------------------------
@dataclass
class MagBetaConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str
    action: str                     # "CLI" or "CLD"

    # data
    action_col: str = "action_id"
    target_col: str = "action_delta_pct"
    max_pct: float = 40.0

    # training
    epochs: int = 25
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda"

    # model
    hidden: int = 256
    depth: int = 3

    # early stopping
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    early_stop_warmup: int = 2

    # dataloader speed
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_target_col(df: pd.DataFrame, target_col: str) -> str:
    if target_col in df.columns:
        return target_col
    if "magnitude_pct" in df.columns:
        print(f"⚠️ target_col '{target_col}' not found. Using 'magnitude_pct' instead.")
        return "magnitude_pct"
    raise RuntimeError(f"Missing target column '{target_col}' and fallback 'magnitude_pct'.")


# -----------------------------
# Core trainer
# -----------------------------
def train_mag_beta(conf: MagBetaConfig) -> Dict[str, Any]:
    set_seed(conf.seed)

    # ---- load data ----
    df_tr = pd.read_parquet(conf.train_parquet)
    df_va = pd.read_parquet(conf.val_parquet)

    action_id = 1 if conf.action.upper() == "CLI" else 2
    df_tr = df_tr[df_tr[conf.action_col].astype(int) == action_id].copy()
    df_va = df_va[df_va[conf.action_col].astype(int) == action_id].copy()

    target_col = resolve_target_col(df_tr, conf.target_col)

    feat_cols = [c for c in df_tr.columns if c.startswith("s_")]
    if not feat_cols:
        raise RuntimeError("No feature columns starting with 's_' found.")

    for c in feat_cols:
        df_tr[c] = pd.to_numeric(df_tr[c], errors="coerce").fillna(0.0)
        df_va[c] = pd.to_numeric(df_va[c], errors="coerce").fillna(0.0)

    y_tr_pp = np.abs(pd.to_numeric(df_tr[target_col], errors="coerce").fillna(0.0).to_numpy())
    y_va_pp = np.abs(pd.to_numeric(df_va[target_col], errors="coerce").fillna(0.0).to_numpy())

    y_tr = np.clip(y_tr_pp / conf.max_pct, 0.0, 1.0).astype(np.float32)
    y_va = np.clip(y_va_pp / conf.max_pct, 0.0, 1.0).astype(np.float32)

    X_tr = df_tr[feat_cols].astype(np.float32).to_numpy()
    X_va = df_va[feat_cols].astype(np.float32).to_numpy()

    mu_np = X_tr.mean(axis=0)
    sd_np = np.where(X_tr.std(axis=0) < 1e-6, 1.0, X_tr.std(axis=0))

    mu = torch.as_tensor(mu_np, device=DEVICE)
    sd = torch.as_tensor(sd_np, device=DEVICE)

    # ---- dataloaders (speed aligned with Gate/DIR) ----
    pin = conf.pin_memory and DEVICE == "cuda"
    nw = conf.num_workers if pin else 0

    train_loader = DataLoader(
        TabDataset(X_tr, y_tr),
        batch_size=conf.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(conf.persistent_workers and nw > 0),
        prefetch_factor=(conf.prefetch_factor if nw > 0 else None),
    )

    val_loader = DataLoader(
        TabDataset(X_va, y_va),
        batch_size=conf.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(conf.persistent_workers and nw > 0),
        prefetch_factor=(conf.prefetch_factor if nw > 0 else None),
    )

    # ---- model ----
    model = BetaRegressor(X_tr.shape[1], conf.hidden, conf.depth).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    best_val = float("inf")
    best_train = float("inf")
    best_epoch = 0
    epochs_since_improve = 0

    Path(conf.out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Training loop
    # -----------------------------
    for ep in range(1, conf.epochs + 1):
        t0 = time.perf_counter()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        train_nlls = []
        n_steps = 0
        n_samples = 0

        for xb, yb in train_loader:
            n_steps += 1
            n_samples += xb.size(0)

            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            xb = torch.nan_to_num(xb)
            xz = (xb - mu) / sd

            mu_p, phi = model(xz)
            nll = beta_nll(yb, mu_p, phi).mean()

            opt.zero_grad(set_to_none=True)
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_nlls.append(float(nll.item()))

        tr_nll = float(np.mean(train_nlls))
        best_train = min(best_train, tr_nll)

        # ---- validation ----
        model.eval()
        val_nlls = []
        mae_pp = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                xb = torch.nan_to_num(xb)
                xz = (xb - mu) / sd

                mu_p, phi = model(xz)
                nll = beta_nll(yb, mu_p, phi).mean()
                val_nlls.append(float(nll.item()))

                pred_pp = mu_p * conf.max_pct
                true_pp = yb * conf.max_pct
                mae_pp.append(float(torch.mean(torch.abs(pred_pp - true_pp)).item()))

        va_nll = float(np.mean(val_nlls))
        va_mae = float(np.mean(mae_pp))

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        samp_s = n_samples / max(1e-9, dt)
        step_s = n_steps / max(1e-9, dt)
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if DEVICE == "cuda" else 0.0

        tag = f"MAG:{conf.action.upper()}"
        print(
            f"[{tag}] Epoch {ep:03d} | "
            f"train_nll={tr_nll:.5f} | val_nll={va_nll:.5f} | val_MAE_pp={va_mae:.3f} | "
            f"{samp_s:,.0f} samp/s {step_s:.2f} step/s {dt:.1f}s"
            + (f" | max_mem={mem_gb:.2f} GB" if DEVICE == "cuda" else "")
        )

        improved = (va_nll < best_val - conf.early_stop_min_delta)
        if improved:
            best_val = va_nll
            best_epoch = ep
            epochs_since_improve = 0

            ckpt = {
                "task": "magnitude_beta",
                "action": conf.action.upper(),
                "model_state": model.state_dict(),
                "feature_cols": feat_cols,
                "scaler_mean": mu_np,
                "scaler_std": sd_np,
                "best_val_nll": best_val,
                "best_train_nll": best_train,
                "best_epoch": best_epoch,
                "config": asdict(conf),
            }
            torch.save(ckpt, conf.out_ckpt)
            print(
                f"✅ Saved BEST checkpoint by val_nll={best_val:.5f} "
                f"(best_train_nll={best_train:.5f}) @epoch={best_epoch} -> {conf.out_ckpt}"
                f"Best val_nll={best_val:.5f} at epoch {best_epoch}."
            )
        else:
            epochs_since_improve += 1

        if ep >= conf.early_stop_warmup and epochs_since_improve >= conf.early_stop_patience:
            print(f"Early stopping. Best val_nll={best_val:.5f} at epoch {best_epoch}.")
            break

    return {
        "out_ckpt": conf.out_ckpt,
        "best_val_nll": best_val,
        "best_train_nll": best_train,
        "best_epoch": best_epoch,
    }
