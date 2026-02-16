#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference_policy.py (v2-compatible)
-----------------------------------
Loads a trained AWAC/AWR Actor-Critic checkpoint produced by:
  train_offline_awac_production_v2.py

Runs ACTOR policy on any trajectories CSV and writes per-row outputs:
- prob_hold/prob_cli/prob_cld
- chosen_action_id + chosen_action_name
- chosen_mag_pct (expected magnitude for CLI/CLD; 0 for HOLD)
- allowed_cli/allowed_cld from guardrails (computed on RAW features)

Usage:
python inference_policy.py \
  --ckpt checkpoints/offline_awac_actor_critic_stable.pt \
  --in_csv rl_dataset/splits/trajectories_test.csv \
  --out_csv rl_dataset/explain/policy_test_preds.csv \
  --batch_size 8192 \
  --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_HOLD, ACTION_CLI, ACTION_CLD = 0, 1, 2


def beta_mean(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return alpha / (alpha + beta)


def compute_action_mask_raw(
    s_raw: torch.Tensor,
    state_cols: list,
    min_score_for_cli: float = 680.0,
    max_dpd_for_cli: float = 1.0,
    min_pay_ratio_for_cli: float = 0.90,
    min_util_for_cli: float = 0.25,
    max_util_for_cli: float = 0.95,
    min_overlimit_for_cld: float = 0.05,
    min_dpd_for_cld: float = 2.0,
    max_score_for_cld: float = 650.0,
) -> torch.Tensor:
    """
    Returns mask [B,3] where True=allowed.
    IMPORTANT: computed on RAW (unstandardized) features so thresholds are meaningful.
    If required columns missing, returns allow-all.
    """
    B = s_raw.shape[0]
    mask = torch.ones((B, 3), dtype=torch.bool, device=s_raw.device)

    def col(name):
        if name in state_cols:
            return s_raw[:, state_cols.index(name)]
        return None

    score = col("s_external_score")
    dpd = col("s_dpd_count_12m")
    pay = col("s_payment_ratio")
    util_max = col("s_max_utilization_6m")
    over = col("s_overlimit_rate_90d")

    if score is None or dpd is None or pay is None or util_max is None:
        return mask

    cli_ok = (
        (score >= min_score_for_cli)
        & (dpd <= max_dpd_for_cli)
        & (pay >= min_pay_ratio_for_cli)
        & (util_max >= min_util_for_cli)
        & (util_max <= max_util_for_cli)
    )

    if over is None:
        cld_ok = (dpd >= min_dpd_for_cld) | (score <= max_score_for_cld)
    else:
        cld_ok = (dpd >= min_dpd_for_cld) | (over >= min_overlimit_for_cld) | (score <= max_score_for_cld)

    mask[:, ACTION_HOLD] = True
    mask[:, ACTION_CLI] = cli_ok
    mask[:, ACTION_CLD] = cld_ok
    return mask


class ActorCritic(nn.Module):
    """
    MUST match train_offline_awac_production_v2.py

    Actor: logits pi(a|s)
    Critic: Q(s,a) for each action
    Baseline: V(s)
    Magnitude: Beta(alpha,beta) for CLI/CLD
    """
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, 3)
        self.q = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)
        self.mag_alpha = nn.Linear(hidden, 1)
        self.mag_beta = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.pi(x)                 # [B,3]
        q = self.q(x)                       # [B,3]
        v = self.v(x).squeeze(-1)           # [B]
        alpha = F.softplus(self.mag_alpha(x)).squeeze(-1) + 1.0
        beta  = F.softplus(self.mag_beta(x)).squeeze(-1) + 1.0
        return logits, q, v, alpha, beta


@torch.no_grad()
def run_inference(
    ckpt_path: str,
    in_csv: str,
    out_csv: str,
    batch_size: int,
    device: str,
):
    # IMPORTANT: v2 bundle is a full dict, so load with weights_only=False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_cols = ckpt["state_cols"]
    obs_dim = int(ckpt["obs_dim"])
    mu = torch.tensor(ckpt["scaler_mean"], dtype=torch.float32, device=device)
    sd = torch.tensor(ckpt["scaler_std"], dtype=torch.float32, device=device)
    mag_max = float(ckpt.get("mag_max", 0.40))

    model = ActorCritic(obs_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    df = pd.read_csv(in_csv, low_memory=False).replace([np.inf, -np.inf], np.nan)

    # Ensure state columns exist
    for c in state_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[state_cols] = df[state_cols].fillna(0.0).astype(np.float32)

    # Keep identifiers if present
    id_cols = []
    for c in ["cust_id", "t_date", "t1_date", "action_id", "magnitude_pct", "reward"]:
        if c in df.columns:
            id_cols.append(c)

    out_rows = []
    n = len(df)

    for i in range(0, n, batch_size):
        chunk = df.iloc[i:i + batch_size].copy()
        s_raw_np = chunk[state_cols].to_numpy(dtype=np.float32, copy=True)

        s_raw = torch.tensor(s_raw_np, device=device)
        s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Guardrails on RAW
        mask = compute_action_mask_raw(s_raw, state_cols)

        # Standardize for model input
        s = (s_raw - mu) / sd
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        logits, q, v, alpha, beta = model(s)

        # Apply mask to logits (disallowed => prob ~ 0)
        logits_masked = logits.masked_fill(~mask, -1e9)

        probs = F.softmax(logits_masked, dim=-1)  # [B,3]
        chosen = torch.argmax(probs, dim=-1)      # [B]

        # Expected magnitude for CLI/CLD
        mag01 = beta_mean(alpha, beta).clamp(0.0, 1.0)
        mag_exp = mag01 * mag_max
        chosen_mag = torch.where(chosen == ACTION_HOLD, torch.zeros_like(mag_exp), mag_exp)

        probs_np = probs.detach().cpu().numpy()
        chosen_np = chosen.detach().cpu().numpy()
        chosen_mag_np = chosen_mag.detach().cpu().numpy()

        allowed_cli = mask[:, ACTION_CLI].detach().cpu().numpy().astype(np.int32)
        allowed_cld = mask[:, ACTION_CLD].detach().cpu().numpy().astype(np.int32)

        # optional: write q and v for debugging/explainability
        q_np = q.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()

        for j in range(len(chunk)):
            row = {c: chunk.iloc[j][c] for c in id_cols}
            row.update({
                "prob_hold": float(probs_np[j, 0]),
                "prob_cli":  float(probs_np[j, 1]),
                "prob_cld":  float(probs_np[j, 2]),
                "chosen_action_id": int(chosen_np[j]),
                "chosen_action_name": ["HOLD", "CLI", "CLD"][int(chosen_np[j])],
                "chosen_mag_pct": float(chosen_mag_np[j]),
                "allowed_cli": int(allowed_cli[j]),
                "allowed_cld": int(allowed_cld[j]),

                # debug signals (useful later for explainability / sanity checks)
                "q_hold": float(q_np[j, 0]),
                "q_cli":  float(q_np[j, 1]),
                "q_cld":  float(q_np[j, 2]),
                "v":      float(v_np[j]),
            })
            out_rows.append(row)

        if (i // batch_size) % 50 == 0:
            print(f"processed {min(i + batch_size, n):,}/{n:,}")

    out_df = pd.DataFrame(out_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved policy inference output: {out_csv} (rows={len(out_df):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
