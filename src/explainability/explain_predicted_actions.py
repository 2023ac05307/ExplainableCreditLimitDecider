#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
explain_selected_customers_shap.py
----------------------------------
Compute SHAP explanations ONLY for selected customers.

Inputs:
  - features parquet (must contain the model feature columns)
  - predictions parquet (must contain next_month_action_3cls OR final_action_3cls)
  - gate_ckpt, dir_ckpt (same ckpts used for prediction)

Outputs:
  - prints a small table (max 20 rows by default)
  - optionally writes an output parquet with explanation columns

Example:
python scripts/explain_selected_customers_shap.py \
  --features_parquet data/staging/trajectory_strict.parquet \
  --pred_parquet data/curated/next_month_prediction \
  --cust_id_col cust_id \
  --cust_ids 10001,10002,10003 \
  --gate_ckpt checkpoints/classification/two_step_gate_checkpoint_subsample.pt \
  --dir_ckpt  checkpoints/classification/two_step_dir_checkpoint_subsample.pt \
  --year 2025 --latest_per_customer \
  --topk 5 --bg 256 --max_print 20 \
  --out_parquet data/curated/explanations_selected
"""

from __future__ import annotations

from IPython.display import display
import argparse
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# SHAP is optional dependency; install if needed:
# pip install shap
import shap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

A_HOLD, A_CLI, A_CLD = 0, 1, 2
ACTION_NAME = {0: "HOLD", 1: "CLI", 2: "CLD"}


# -----------------------------
# Simple I/O (parquet file/dir)
# -----------------------------
def load_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_parquet(p)
    return df.replace([np.inf, -np.inf], np.nan)

def save_parquet_any(df: pd.DataFrame, out_path: str) -> str:
    outp = Path(out_path)
    if outp.suffix.lower() == ".parquet":
        outp.parent.mkdir(parents=True, exist_ok=True)
        tmp = outp.with_suffix(outp.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, outp)
        return str(outp)
    outp.mkdir(parents=True, exist_ok=True)
    part = outp / f"part-{uuid.uuid4().hex}.parquet"
    df.to_parquet(part, index=False)
    return str(outp)


# -----------------------------
# Models (match your trainers)
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
# CKPT loaders (trusted)
# -----------------------------
def load_gate(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = list(ck["state_cols"])
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32)

    m = ActorCriticBinary(obs_dim).to(DEVICE)
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    return m, state_cols, mu, sd

def load_dir(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_dim = int(ck["obs_dim"])
    state_cols = list(ck["state_cols"])
    mu = np.asarray(ck["scaler_mean"], dtype=np.float32)
    sd = np.asarray(ck["scaler_std"], dtype=np.float32)

    sdict = ck["model"]
    has_mag = any(k.startswith("mag_alpha") or k.startswith("mag_beta") for k in sdict.keys())

    if has_mag:
        m = ActorCriticDirBinary(obs_dim).to(DEVICE)
        m.load_state_dict(sdict, strict=True)
    else:
        m = ActorCriticDirBinaryNoMag(obs_dim).to(DEVICE)
        m.load_state_dict(sdict, strict=True)

    m.eval()
    return m, state_cols, mu, sd, has_mag


# -----------------------------
# Feature prep
# -----------------------------
def ensure_cols(df: pd.DataFrame, cols: List[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ctx} Missing cols: {missing[:25]}{'...' if len(missing)>25 else ''}")

def to_f32(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.astype(np.float32).to_numpy(copy=True)

def normalize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    sd = np.clip(sd, 1e-6, None)
    return (X - mu[None, :]) / sd[None, :]


# -----------------------------
# SHAP wrappers
# -----------------------------
class GateProbWrapper(nn.Module):
    def __init__(self, gate_model: nn.Module):
        super().__init__()
        self.m = gate_model
    def forward(self, x):
        logit, _, _ = self.m(x)
        return torch.sigmoid(logit).unsqueeze(-1)

class DirProbWrapper(nn.Module):
    def __init__(self, dir_model: nn.Module, has_mag: bool):
        super().__init__()
        self.m = dir_model
        self.has_mag = has_mag
    def forward(self, x):
        if self.has_mag:
            logit, _, _, _, _ = self.m(x)
        else:
            logit, _, _ = self.m(x)
        return torch.sigmoid(logit).unsqueeze(-1)


class DirLogitOnly(torch.nn.Module):
    def __init__(self, dir_model, has_mag: bool):
        super().__init__()
        self.dir = dir_model
        self.has_mag = has_mag

    def forward(self, x):
        if self.has_mag:
            logit, _, _, _, _ = self.dir(x)   # (N,)
        else:
            logit, _, _ = self.dir(x)         # (N,)
        return logit.unsqueeze(-1)            # (N,1)


# def topk_shap(shap_row, feat_names, k):
#     v = np.asarray(shap_row).reshape(-1)   # <-- critical
#     idx = np.argsort(np.abs(v))[::-1][:k]
#     feats = [feat_names[int(j)] for j in idx]
#     vals  = [float(v[int(j)]) for j in idx]
#     return feats, vals

def topk_shap(shap_row, feat_names, k: int):
    """
    Returns:
      feats_top: list[str]
      vals_top:  list[float]  (same order)
    Works whether shap_row is shape (d,) or (1,d) etc.
    """
    sv = np.asarray(shap_row)
    sv = sv.reshape(-1)  # flatten

    k = min(k, sv.size)
    idx = np.argsort(np.abs(sv))[-k:][::-1]  # top |shap|
    feats_top = [feat_names[int(i)] for i in idx]
    vals_top = [float(sv[int(i)]) for i in idx]
    return feats_top, vals_top


def normalize_shap(shap_vals):
    # SHAP sometimes returns a list even for single output: [N,F]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    shap_vals = np.asarray(shap_vals)

    # if (N,F,1) -> (N,F)
    if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
        shap_vals = shap_vals[..., 0]

    # if (N,1,F) -> (N,F)
    if shap_vals.ndim == 3 and shap_vals.shape[1] == 1:
        shap_vals = shap_vals[:, 0, :]

    return shap_vals



def action_name(a: int) -> str:
    return "HOLD" if a == A_HOLD else ("CLI" if a == A_CLI else "CLD")


# You can customize this mapping into customer-friendly reasons
FEATURE_REASON_MAP = {
    "s_payment_ratio": "repayment strength",
    "s_min_pay_ratio": "minimum payment behavior",
    "s_utilization": "credit utilization",
    "s_external_score": "bureau score",
    "s_external_score_delta": "recent bureau score change",
    "s_dpd_flag": "delinquency signal",
    "s_balance_to_income": "balance-to-income pressure",
}

FEATURE_PHRASE = {
  "s_credit_limit_max_6m": "your credit limit has been relatively high in the last 6 months",
  "s_credit_limit_mean_3m": "your average credit limit over the last 3 months",
  "s_credit_limit_mean_12m": "your average credit limit over the last 12 months",
  "s_credit_limit_trend_6m": "the recent trend in your credit limit over the last 6 months",
  "s_credit_limit_last_vs_mean_6m": "your latest credit limit compared to your 6-month average",
  "s_credit_limit_max_3m": "your peak credit limit in the last 3 months",
  "s_credit_limit_min_3m": "your lowest credit limit in the last 3 months",

  "s_limit_to_income_mean_3m": "your limit-to-income profile over the last 3 months",
  "s_limit_to_income_max_12m": "your highest limit-to-income level in the last 12 months",
  "s_limit_to_income_min_6m": "your lowest limit-to-income level in the last 6 months",
  "s_limit_to_income_last_vs_mean_6m": "your recent limit-to-income level versus your 6-month average",

  "s_cli_count_12m": "how often your limit was increased in the last 12 months",
  "s_cld_count_12m": "how often your limit was decreased in the last 12 months",

  # add more over time…
}


FEATURE_PHRASE = {
    "s_credit_limit_max_6m": "your credit limit was relatively high recently",
    "s_credit_limit_mean_3m": "your average credit limit over the last 3 months",
    "s_credit_limit_trend_6m": "the recent trend in your credit limit",
    "s_limit_to_income_max_12m": "your limit relative to your income over the last year",
    "s_limit_to_income_mean_3m": "your recent limit-to-income profile",
    "s_cli_count_12m": "how often your limit was increased in the past year",
    "s_cld_count_12m": "how often your limit was reduced in the past year",
    # keep expanding this dictionary over time
}


def humanize_feature(f: str) -> str:
    if f in FEATURE_PHRASE:
        return FEATURE_PHRASE[f]

    t = f.replace("s_", "")
    t = t.replace("_", " ")

    # fix common words
    t = t.replace("mean", "average")
    t = t.replace("max", "highest")
    t = t.replace("min", "lowest")
    t = t.replace("trend", "trend in")

    # time windows
    t = t.replace("3m", "the last three months")
    t = t.replace("6m", "the last six months")
    t = t.replace("12m", "the past year")

    # clean grammar
    t = t.replace("last vs mean", "recent level compared to average")
    t = t.replace("limit to income", "limit-to-income")

    return "your " + t


def make_natural_explanation(action_name: str, reason_features_csv: str) -> str:
    feats = [x.strip() for x in (reason_features_csv or "").split(",") if x.strip()]
    feats = feats[:5]
    phrases = [humanize_feature(f) for f in feats]

    if action_name == "CLI":
        if phrases:
            return ("We increased your credit limit mainly because " +
                    ", ".join(phrases[:-1]) + (", and " + phrases[-1] if len(phrases) > 1 else phrases[0]) +
                    ".")
        return "We increased your credit limit based on your recent account profile."
    elif action_name == "CLD":
        if phrases:
            return ("We reduced your credit limit mainly because " +
                    ", ".join(phrases[:-1]) + (", and " + phrases[-1] if len(phrases) > 1 else phrases[0]) +
                    ".")
        return "We reduced your credit limit based on your recent account profile."
    else:  # HOLD
        # HOLD explanations should not list too many “reasons”; keep it reassuring.
        return "We kept your credit limit unchanged because your recent account behavior looks stable and did not require an adjustment."


def build_explanation(action_name: str, feats_top: list[str], vals_top: list[float]) -> str:
    """
    More natural, sign-aware explanation.

    - Uses only positive SHAP contributors for the predicted action (fallback to all if none).
    - Produces complete, customer-friendly sentences with proper conjunctions.
    - Adds an ending clause to avoid "comma-fragment" sentences.
    """

    # Pair and keep the contributors
    pairs = [(f, float(v)) for f, v in zip(feats_top, vals_top)]

    # Prefer positive contributors for the predicted action
    pos = [(f, v) for f, v in pairs if v > 0]
    if not pos:
        pos = pairs  # fallback

    # Humanize feature names
    phrases = [humanize_feature(f) for f, _ in pos[:5]]

    # Utility: join like "A", "A and B", "A, B, and C"
    def join_natural(items: list[str]) -> str:
        items = [s.strip().rstrip(".") for s in items if s and str(s).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    # HOLD stays simple & safe
    if action_name == "HOLD":
        return "We kept your credit limit unchanged because your recent account indicators looked stable overall."

    # Action verb + closing rationale clause (helps semantics)
    verb = "increased" if action_name == "CLI" else "reduced"
    tail = "showed improved account strength" if action_name == "CLI" else "indicated increased risk"

    reasons = join_natural(phrases)

    # If reasons are empty, still return something sensible
    if not reasons:
        return f"We {verb} your credit limit because your recent account indicators suggested this change."

    return f"We {verb} your credit limit mainly because {reasons} {tail}."




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_parquet", required=True, help="Parquet with s_* features")
    ap.add_argument("--pred_parquet", required=True, help="Predictions parquet (contains next_month_action_3cls or final_action_3cls)")
    ap.add_argument("--cust_id_col", default="cust_id")
    ap.add_argument("--cust_ids", required=True, help="Comma-separated customer ids to explain")
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--latest_per_customer", action="store_true")
    ap.add_argument("--t_date_col", default="t_date", help="Date column to use for 'latest per customer' (must be parseable)")
    ap.add_argument("--out_html", default="", help="Optional: write a pretty HTML preview table.")
    ap.add_argument("--gate_ckpt", required=True)
    ap.add_argument("--dir_ckpt", required=True)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--bg", type=int, default=256, help="Background sample size for SHAP")
    ap.add_argument("--max_print", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_parquet", default="", help="Optional: save explanation rows to parquet")
    ap.add_argument("--next_month", default="2026-01", help="Next month label to stamp in output table.")
    ap.add_argument("--prev_limit_col", default="s_credit_limit", help="Column holding previous month credit limit.")
    ap.add_argument("--mag_pct_col", default="next_month_magnitude_pct_points", help="Magnitude percent points column from prediction output.")
    ap.add_argument("--mag_frac_col", default="next_month_magnitude_frac", help="Magnitude frac column fallback.")
    ap.add_argument("--mag_clip_min", type=float, default=0.0)
    ap.add_argument("--mag_clip_max", type=float, default=40.0)
    ap.add_argument("--out_customer_table", default="", help="Output parquet path/dir for final customer table with explanation.")

    args = ap.parse_args()

    # ---- Load ----
    feats = load_parquet_any(args.features_parquet)
    preds = load_parquet_any(args.pred_parquet)

    # ---- Align/filter: year + latest ----
    # Year filter expects t_date-like column exists in feats
    if args.t_date_col in feats.columns:
        d = pd.to_datetime(feats[args.t_date_col], errors="coerce")
        feats = feats.loc[d.dt.year == args.year].copy()

        if args.latest_per_customer:
            # latest row per customer
            feats["_dt"] = d.loc[feats.index]
            feats = feats.sort_values([args.cust_id_col, "_dt"]).groupby(args.cust_id_col, as_index=False).tail(1)
            feats = feats.drop(columns=["_dt"], errors="ignore")

        #print(f"[INFO] After year filter: rows={len(feats)} | year={args.year} | latest_per_customer={args.latest_per_customer}")

    # ---- Filter to customers ----
    cust_list = [c.strip() for c in args.cust_ids.split(",") if c.strip()]
    feats = feats[feats[args.cust_id_col].astype(str).isin(set(cust_list))].copy()
    preds = preds[preds[args.cust_id_col].astype(str).isin(set(cust_list))].copy()
    preds = preds.drop_duplicates(subset=[args.cust_id_col], keep="last")
    feats = feats.drop_duplicates(subset=[args.cust_id_col], keep="last")
    if len(feats) == 0 or len(preds) == 0:
        raise RuntimeError("No rows found for the given cust_ids after filtering.")

    # ---- Merge features + prediction action ----
    # Prefer next_month_action_3cls if present
    action_col = "next_month_action_3cls" if "next_month_action_3cls" in preds.columns else "final_action_3cls"
    if action_col not in preds.columns:
        raise RuntimeError("Prediction file must contain next_month_action_3cls or final_action_3cls.")

    pred_keep = [args.cust_id_col, action_col]
    if args.mag_pct_col in preds.columns:
        pred_keep.append(args.mag_pct_col)
    if args.mag_frac_col in preds.columns:
        pred_keep.append(args.mag_frac_col)

    merged = feats.merge(preds[pred_keep], on=args.cust_id_col, how="inner")
    if len(merged) == 0:
        raise RuntimeError("No matching rows after joining features with predictions (cust_id mismatch).")

    # ---- Load models ----
    gate_m, gate_cols, gmu, gsd = load_gate(args.gate_ckpt)
    dir_m, dir_cols, dmu, dsd, dir_has_mag = load_dir(args.dir_ckpt)

    ensure_cols(merged, gate_cols, "Gate:")
    ensure_cols(merged, dir_cols, "Dir:")

    Xg = normalize(to_f32(merged, gate_cols), gmu, gsd)
    Xd = normalize(to_f32(merged, dir_cols), dmu, dsd)

    # ---- Background sample comes from full-year features (better than tiny selection)
    bg_src = load_parquet_any(args.features_parquet)
    if args.t_date_col in bg_src.columns:
        dt = pd.to_datetime(bg_src[args.t_date_col], errors="coerce")
        bg_src = bg_src.loc[dt.dt.year == args.year].copy()
    if args.latest_per_customer and args.cust_id_col in bg_src.columns and args.t_date_col in bg_src.columns:
        bg_src["_dt"] = pd.to_datetime(bg_src[args.t_date_col], errors="coerce")
        bg_src = bg_src.sort_values([args.cust_id_col, "_dt"]).groupby(args.cust_id_col, as_index=False).tail(1)
        bg_src = bg_src.drop(columns=["_dt"], errors="ignore")

    # Prepare background arrays
    ensure_cols(bg_src, gate_cols, "BG Gate:")
    ensure_cols(bg_src, dir_cols, "BG Dir:")
    Xg_bg = normalize(to_f32(bg_src, gate_cols), gmu, gsd)
    Xd_bg = normalize(to_f32(bg_src, dir_cols), dmu, dsd)

    rng = np.random.default_rng(args.seed)
    bg_n = min(args.bg, len(bg_src))
    bg_idx = rng.choice(len(bg_src), size=bg_n, replace=False)

    # ---- SHAP explain (FIXED) ----
    gate_wrap = GateProbWrapper(gate_m).to(DEVICE).eval()
    dir_wrap_logit = DirLogitOnly(dir_m, dir_has_mag).to(DEVICE).eval()

    # tensors
    Xg_t = torch.tensor(Xg, dtype=torch.float32, device=DEVICE)
    Xd_t = torch.tensor(Xd, dtype=torch.float32, device=DEVICE)

    bg_gate_t = torch.tensor(Xg_bg[bg_idx], dtype=torch.float32, device=DEVICE)
    bg_dir_t  = torch.tensor(Xd_bg[bg_idx], dtype=torch.float32, device=DEVICE)

    # GradientExplainer for gate probability (sigmoid(logit))
    expl_gate = shap.GradientExplainer(gate_wrap, bg_gate_t)
    shap_gate = normalize_shap(expl_gate.shap_values(Xg_t))

    # GradientExplainer for dir LOGIT (more stable than prob)
    expl_dir = shap.GradientExplainer(dir_wrap_logit, bg_dir_t)
    shap_dir  = normalize_shap(expl_dir.shap_values(Xd_t))

    # shap can return list for single output; normalize to array
    if isinstance(shap_gate, list):
        shap_gate = shap_gate[0]
    if isinstance(shap_dir, list):
        shap_dir = shap_dir[0]

    shap_gate = np.asarray(shap_gate)  # shape: (N, D_gate)
    shap_dir  = np.asarray(shap_dir)   # shape: (N, D_dir)


    # ---- Build explanations per row based on predicted action ----
    exp_text = []
    exp_feats = []

    for i in range(len(merged)):
        a = int(merged[action_col].iloc[i])
        a_name = action_name(a)

        if a_name == "HOLD":
            feats_top, vals_top = topk_shap(shap_gate[i], gate_cols, args.topk)
        else:
            feats_top, vals_top = topk_shap(shap_dir[i], dir_cols, args.topk)

        exp_text.append(build_explanation(a_name, feats_top, vals_top))
        exp_feats.append(",".join(feats_top))

    merged["action_name"] = merged[action_col].astype(int).map({0:"HOLD",1:"CLI",2:"CLD"})
    merged["reason_features"] = ""
    #exp_feats
    merged["explanation"] = exp_text


    # -----------------------------
    # Build Customer Table + Explanation
    # -----------------------------
    if args.prev_limit_col not in merged.columns:
        raise RuntimeError(f"prev_limit_col '{args.prev_limit_col}' not found in merged dataframe.")

    # action id/name
    action_id = pd.to_numeric(merged[action_col], errors="coerce").fillna(0).astype(int)
    action_taken = action_id.map(ACTION_NAME).fillna("HOLD")

    # magnitude % points (prefer pct_points; fallback to frac*100; else 0)
    if args.mag_pct_col in merged.columns:
        mag_pct = pd.to_numeric(merged[args.mag_pct_col], errors="coerce").fillna(0.0)
    elif args.mag_frac_col in merged.columns:
        mag_pct = pd.to_numeric(merged[args.mag_frac_col], errors="coerce").fillna(0.0) * 100.0
    else:
        mag_pct = pd.Series(0.0, index=merged.index)

    mag_pct = mag_pct.clip(args.mag_clip_min, args.mag_clip_max)

    # HOLD => magnitude 0
    mag_pct_effective = mag_pct.where(action_id != A_HOLD, 0.0)

    # prev limit
    prev_limit = pd.to_numeric(merged[args.prev_limit_col], errors="coerce").fillna(0.0)

    # updated limit
    mult = np.ones(len(merged), dtype=np.float32)
    cli_mask = (action_id == A_CLI).to_numpy()
    cld_mask = (action_id == A_CLD).to_numpy()
    mult[cli_mask] = 1.0 + (mag_pct_effective[cli_mask].to_numpy() / 100.0)
    mult[cld_mask] = 1.0 - (mag_pct_effective[cld_mask].to_numpy() / 100.0)

    updated_limit = np.round(prev_limit.to_numpy() * mult, 2)

    # Final customer table
    customer_table = pd.DataFrame({
        args.cust_id_col: merged[args.cust_id_col],
        "next_month": args.next_month,
        "action_taken": action_taken,
        "magnitude_percentage": np.round(mag_pct_effective.to_numpy(), 3),
        "prev_credit_limit": np.round(prev_limit.to_numpy(), 2),
        "updated_credit_limit": updated_limit,
        "explanation": merged["explanation"].fillna(""),
    })

    # -----------------------------
    # Print max 20 (table-like)
    # -----------------------------
    #print("\n=== Customer Table with Explanation (max 20) ===")
    #print(customer_table.head(args.max_print).to_string(index=False))
    # print("\n=== Customer Table with Explanation ===")
    # print(
    #     customer_table.head()
    #     .to_string(index=False, max_colwidth=90)  # keeps explanation readable
    # )

    # -----------------------------
    # Print table in html format (optional)
    # -----------------------------

    if args.out_html:
        html_path = Path(args.out_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)

        sty = (
            customer_table.head(args.max_print).style
                .hide(axis="index")
                .set_properties(subset=["explanation"], **{"white-space": "pre-wrap", "max-width": "700px"})
                .format({
                    "magnitude_percentage": "{:.3f}",
                    "prev_credit_limit": "{:,.2f}",
                    "updated_credit_limit": "{:,.2f}",
                })
        )

        html_path.write_text(sty.to_html(), encoding="utf-8")
        print(f"**Wrote HTML preview: {html_path}")


    # -----------------------------
    # Save parquet (optional)
    # -----------------------------
    if args.out_customer_table:
        saved = save_parquet_any(customer_table.head(args.max_print), args.out_customer_table)
        print(f"\n***Saved customer table with explanation to: {saved}")






if __name__ == "__main__":
    main()
