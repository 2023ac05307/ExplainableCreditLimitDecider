#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
explain_two_step_shap_reasoncodes.py
-----------------------------------
SOTA explainability for your 2-step pipeline (Gate + Dir) using SHAP.

What it does:
1) Reads an inference output CSV (must contain s_* features + predictions).
2) Loads gate_ckpt and dir_ckpt (same ones used for inference).
3) Computes SHAP values for:
   - Gate: P(NON_HOLD)
   - Dir : P(CLI)    (meaningful only when predicted NON_HOLD)
4) Converts top SHAP drivers into "Reason Codes" using a JSON mapping.
5) Writes a new CSV with a single customer-facing explanation column:
   - explanation

Notes:
- This is designed to run AFTER:
  - infer_two_step_gate_dir.py  OR
  - predict_next_month_action_and_limit.py
- Your inference CSV should still include the input snapshot feature columns.
- SHAP on a neural net can be expensive; we:
  - use a small background sample
  - compute explanations in batches

Install:
pip install shap

Usage:
python explain_two_step_shap_reasoncodes.py \
  --in_csv out/two_step_preds.csv \
  --gate_ckpt checkpoints/gate_awac_binary_bce_best.pt \
  --dir_ckpt  checkpoints/dir_awac_best.pt \
  --reason_json reason_codes.json \
  --out_csv out/two_step_preds_with_explanations.csv \
  --bg_rows 512 \
  --max_rows 50000 \
  --batch_size 2048
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Human note:
# - We use SHAP GradientExplainer for PyTorch MLPs.
# - DeepExplainer can be finicky across torch versions; GradientExplainer is typically safer.
# -----------------------------
try:
    import shap
except Exception as e:
    raise RuntimeError(
        "Missing dependency: shap. Install with: pip install shap\n"
        f"Original error: {e}"
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

A_HOLD, A_CLI, A_CLD = 0, 1, 2


# -----------------------------
# Human note: basic safety helpers
# -----------------------------
def ensure_cols(df: pd.DataFrame, cols: list[str], name: str = "required"):
    """Fail fast if the CSV schema drifted (prevents silent wrong explanations)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing {name} columns: {missing[:25]}{'...' if len(missing) > 25 else ''}")


def to_float32_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Human note:
    - In your training scripts you did numeric coercion + NaN fill.
    - We replicate that here to keep SHAP consistent with inference/training.
    """
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32).to_numpy(copy=True)


def standardize_np(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Human note: checkpoint scaler must be applied exactly like inference."""
    sd2 = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return ((X - mu) / sd2).astype(np.float32)


def pick_background(X: np.ndarray, bg_rows: int, seed: int = 42) -> np.ndarray:
    """Human note: background set is critical for stable SHAP values."""
    n = X.shape[0]
    if n <= bg_rows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=bg_rows, replace=False)
    return X[idx]


# -----------------------------
# Model definitions (must match training)
# -----------------------------
class ActorCriticBinary(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
# Checkpoint loaders
# -----------------------------
def load_gate(ckpt_path: str):
    """Human note: gate ckpt contains state_cols + scaler stats + model weights."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)
    model = ActorCriticBinary(obs_dim).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cols, mu, sd


def load_dir(ckpt_path: str):
    """Human note: dir ckpt may contain magnitude heads; we auto-detect."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    cols = ck["state_cols"]
    mu = ck["scaler_mean"].astype(np.float32)
    sd = ck["scaler_std"].astype(np.float32)

    sdict = ck["model"]
    has_mag = any(k.startswith("mag_alpha.") or k.startswith("mag_beta.") for k in sdict.keys())
    if has_mag:
        model = ActorCriticDirBinary(obs_dim).to(DEVICE)
    else:
        model = ActorCriticDirBinaryNoMag(obs_dim).to(DEVICE)

    model.load_state_dict(sdict, strict=True)
    model.eval()
    return model, cols, mu, sd


# -----------------------------
# SHAP wrappers (probability outputs)
# -----------------------------
class ProbWrapper(nn.Module):
    """
    Human note:
    - SHAP expects a function mapping X -> output.
    - We feed it standardized X and output probability of class=1.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        # Gate & Dir both return logit first in tuple
        logit = out[0]
        return torch.sigmoid(logit).unsqueeze(-1)  # (B,1)


def shap_values_in_batches(explainer, X: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Human note:
    - SHAP can be memory heavy.
    - We compute in batches and concat.
    """
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(DEVICE)
        sv = explainer.shap_values(xb)
        # For single output: shap_values returns array-like; normalize it
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        # sv shape expected: (B, D, 1) or (B, D)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        outs.append(sv.astype(np.float32))
    return np.concatenate(outs, axis=0)


# -----------------------------
# Reason-code mapping + explanation text
# -----------------------------
def load_reason_json(path: str) -> dict:
    """Human note: keep reason mapping versioned and auditable."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def topk_feature_reasons(feature_names: list[str], shap_row: np.ndarray, reason_cfg: dict,
                         want_positive: bool, top_k: int) -> list[str]:
    """
    Human note:
    - want_positive=True: pick top positive contributors
    - want_positive=False: pick top negative contributors
    - if not enough, fallback to absolute top-k.
    """
    fmap = reason_cfg.get("feature_map", {})
    vals = shap_row.copy()

    # Select candidates by sign
    if want_positive:
        idx = np.where(vals > 0)[0]
    else:
        idx = np.where(vals < 0)[0]

    if idx.size == 0:
        # fallback: absolute strongest
        idx_sorted = np.argsort(-np.abs(vals))[:top_k]
    else:
        idx_sorted = idx[np.argsort(-np.abs(vals[idx]))][:top_k]

    reasons = []
    for j in idx_sorted.tolist():
        feat = feature_names[j]
        meta = fmap.get(feat)

        if meta is None:
            # fallback: readable feature name
            reasons.append(feat.replace("s_", "").replace("_", " "))
            continue

        if want_positive:
            reasons.append(meta.get("positive", meta.get("reason_code", feat)))
        else:
            reasons.append(meta.get("negative", meta.get("reason_code", feat)))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for r in reasons:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq[:top_k]


def build_customer_explanation(action_3cls: int, reasons: list[str], reason_cfg: dict) -> str:
    """Human note: customer statements should be short and non-technical."""
    templates = reason_cfg.get("templates", {})
    fallbacks = reason_cfg.get("fallback_reasons", {})

    if action_3cls == A_CLI:
        key = "CLI"
    elif action_3cls == A_CLD:
        key = "CLD"
    else:
        key = "HOLD"

    if not reasons:
        reasons = fallbacks.get(key, ["recent account signals"])

    # Make it nice: "A, B, and C"
    if len(reasons) == 1:
        phrase = reasons[0]
    elif len(reasons) == 2:
        phrase = f"{reasons[0]} and {reasons[1]}"
    else:
        phrase = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    tmpl = templates.get(key, "{reasons}.")
    return tmpl.format(reasons=phrase)


def combine_gate_dir_reasons(final_action: int,
                             gate_reasons_pos: list[str], gate_reasons_neg: list[str],
                             dir_reasons_pos: list[str], dir_reasons_neg: list[str],
                             top_k: int) -> list[str]:
    """
    Human note:
    - CLI: Gate says "change is needed" (positive toward NON_HOLD) + Dir says "increase" (positive toward CLI)
    - CLD: Gate positive + Dir negative (toward CLD)
    - HOLD: Gate negative (toward HOLD)
    """
    if final_action == A_CLI:
        reasons = gate_reasons_pos + dir_reasons_pos
    elif final_action == A_CLD:
        reasons = gate_reasons_pos + dir_reasons_neg
    else:
        reasons = gate_reasons_neg

    # Deduplicate + keep only top_k
    seen = set()
    out = []
    for r in reasons:
        if r not in seen:
            out.append(r)
            seen.add(r)
        if len(out) >= top_k:
            break
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Inference output CSV (must contain s_* + predictions)")
    ap.add_argument("--gate_ckpt", required=True)
    ap.add_argument("--dir_ckpt", required=True)
    ap.add_argument("--reason_json", required=True, help="Reason-code mapping JSON")
    ap.add_argument("--out_csv", required=True)

    ap.add_argument("--id_col", default="cust_id")
    ap.add_argument("--action_col", default="final_action_3cls",
                    help="3-class predicted action column (0=HOLD,1=CLI,2=CLD). "
                         "If your file uses next_month_action_3cls, pass that.")
    ap.add_argument("--bg_rows", type=int, default=512, help="SHAP background rows (tradeoff speed vs stability)")
    ap.add_argument("--max_rows", type=int, default=0, help="If >0, explain only first N rows (debug)")
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--write_debug", action="store_true",
                    help="If set, write debug columns with top features for gate/dir.")
    args = ap.parse_args()

    reason_cfg = load_reason_json(args.reason_json)
    top_k = int(reason_cfg.get("top_k", 3))

    df = pd.read_csv(args.in_csv, low_memory=False).replace([np.inf, -np.inf], np.nan)

    ensure_cols(df, [args.id_col], name="id")
    ensure_cols(df, [args.action_col], name="predicted action")

    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows)).copy()

    # ---- Load models + training feature schema
    gate_model, gate_cols, gate_mu, gate_sd = load_gate(args.gate_ckpt)
    dir_model, dir_cols, dir_mu, dir_sd = load_dir(args.dir_ckpt)

    ensure_cols(df, gate_cols, name="gate feature")
    ensure_cols(df, dir_cols, name="dir feature")

    # ---- Build standardized feature matrices (same as inference)
    Xg_raw = to_float32_matrix(df, gate_cols)
    Xd_raw = to_float32_matrix(df, dir_cols)

    Xg = standardize_np(Xg_raw, gate_mu, gate_sd)
    Xd = standardize_np(Xd_raw, dir_mu, dir_sd)

    # ---- Pick SHAP background
    Xg_bg = pick_background(Xg, args.bg_rows, seed=args.seed)
    Xd_bg = pick_background(Xd, args.bg_rows, seed=args.seed)

    # ---- Create SHAP explainers
    # Human note: GradientExplainer requires torch tensors.
    gate_wrap = ProbWrapper(gate_model).to(DEVICE)
    dir_wrap = ProbWrapper(dir_model).to(DEVICE)

    gate_expl = shap.GradientExplainer(gate_wrap, torch.from_numpy(Xg_bg).float().to(DEVICE))
    dir_expl  = shap.GradientExplainer(dir_wrap,  torch.from_numpy(Xd_bg).float().to(DEVICE))

    # ---- Compute SHAP values (batch)
    gate_shap = shap_values_in_batches(gate_expl, Xg, batch_size=args.batch_size)  # (N, Dg)
    dir_shap  = shap_values_in_batches(dir_expl,  Xd, batch_size=args.batch_size)  # (N, Dd)

    # ---- Build explanations row-by-row
    final_action = df[args.action_col].astype(int).to_numpy()
    explanations = []

    # Optional debug fields
    dbg_gate = []
    dbg_dir = []

    for i in range(len(df)):
        # Gate output is P(NON_HOLD)
        # Positive SHAP -> pushes toward NON_HOLD; Negative -> pushes toward HOLD
        gate_pos = topk_feature_reasons(gate_cols, gate_shap[i], reason_cfg, want_positive=True, top_k=top_k)
        gate_neg = topk_feature_reasons(gate_cols, gate_shap[i], reason_cfg, want_positive=False, top_k=top_k)

        # Dir output is P(CLI) among non-hold context
        # Positive SHAP -> pushes toward CLI; Negative -> pushes toward CLD
        dir_pos = topk_feature_reasons(dir_cols, dir_shap[i], reason_cfg, want_positive=True, top_k=top_k)
        dir_neg = topk_feature_reasons(dir_cols, dir_shap[i], reason_cfg, want_positive=False, top_k=top_k)

        reasons = combine_gate_dir_reasons(
            final_action=final_action[i],
            gate_reasons_pos=gate_pos,
            gate_reasons_neg=gate_neg,
            dir_reasons_pos=dir_pos,
            dir_reasons_neg=dir_neg,
            top_k=top_k
        )

        explanations.append(build_customer_explanation(final_action[i], reasons, reason_cfg))

        if args.write_debug:
            dbg_gate.append({"gate_pos": gate_pos, "gate_neg": gate_neg})
            dbg_dir.append({"dir_pos": dir_pos, "dir_neg": dir_neg})

    out = df.copy()
    out["explanation"] = explanations

    if args.write_debug:
        out["debug_gate_reasons"] = [json.dumps(x, ensure_ascii=False) for x in dbg_gate]
        out["debug_dir_reasons"]  = [json.dumps(x, ensure_ascii=False) for x in dbg_dir]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("✅ Saved explanations CSV:", args.out_csv)
    print("Rows:", len(out))
    print("Example explanation:", out["explanation"].iloc[0])


if __name__ == "__main__":
    main()
