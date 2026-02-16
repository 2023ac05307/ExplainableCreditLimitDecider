#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_engineering_monthly_only.py
----------------------------------
INPUT : <in_dir>/snapshots_base_YYYY-MM.parquet   (monthly base snapshots)
OUTPUT: <out_dir>/snapshots_YYYY-MM.parquet       (monthly engineered snapshots)

Monthly-only. No yearly/all_years outputs.

Design:
- For each target month YYYY-MM:
  - load lookback window of base snapshots (default 12 months) + current month
  - run add_rolling_stats() on the window (same logic as your old all_years script)
  - slice only rows for the current month
  - write snapshots_YYYY-MM.parquet
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

STATUS_TO_ID = {"current": 0, "dpd30": 1, "dpd60": 2, "dpd90": 3, "default": 4, "closed": 5}
REGIME_TO_ID = {"boom": 0, "normal": 1, "recession": 2}

WINDOWS = (3, 6, 12)
EPS = 1e-6
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def _require_pyarrow():
    try:
        import pyarrow  # noqa
    except Exception as e:
        raise RuntimeError("Install pyarrow: pip install pyarrow") from e


def write_parquet(df: pd.DataFrame, path: str):
    _require_pyarrow()
    df.to_parquet(path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)


def _slope_apply_factory():
    def _slope(y: np.ndarray):
        y = np.asarray(y, dtype=np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        n = y.shape[0]
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float32)
        x = x - x.mean()
        denom = float((x * x).sum()) + 1e-12
        return float((x * (y - y.mean())).sum() / denom)
    return _slope


# -----------------------------
# YOUR ORIGINAL add_rolling_stats LOGIC (kept same)
# -----------------------------
def add_rolling_stats(snap: pd.DataFrame, group_key="cust_id", date_col="statement_date"):
    snap = snap.sort_values([group_key, date_col]).reset_index(drop=True)

    base_numeric = [
        "payment_ratio", "utilization", "balance", "credit_limit",
        "min_pay_ratio", "min_due",
        "tx_count_30d", "avg_tx_amt_90d", "overlimit_rate_90d",
        "external_score", "external_score_delta",
        "monthly_income",
        "dpd_count_12m", "max_utilization_6m",
        "recent_cli_effectiveness",
    ]
    for c in base_numeric:
        if c in snap.columns:
            snap[c] = pd.to_numeric(snap[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    derived_cols = {}
    if "monthly_income" in snap.columns:
        if "balance" in snap.columns:
            derived_cols["balance_to_income"] = (snap["balance"] / (snap["monthly_income"] + EPS)).astype(np.float32)
        if "credit_limit" in snap.columns:
            derived_cols["limit_to_income"] = (snap["credit_limit"] / (snap["monthly_income"] + EPS)).astype(np.float32)

    if "status_id" in snap.columns:
        derived_cols["dpd_flag"] = (snap["status_id"] >= STATUS_TO_ID["dpd30"]).astype(np.int32)
        derived_cols["severe_dpd_flag"] = (snap["status_id"] >= STATUS_TO_ID["dpd90"]).astype(np.int32)

    if "action_id" in snap.columns:
        derived_cols["is_cli"] = (snap["action_id"] == 1).astype(np.int32)
        derived_cols["is_cld"] = (snap["action_id"] == 2).astype(np.int32)

    if "macro_regime_id" in snap.columns:
        derived_cols["is_recession"] = (snap["macro_regime_id"] == REGIME_TO_ID["recession"]).astype(np.int32)
        derived_cols["is_boom"] = (snap["macro_regime_id"] == REGIME_TO_ID["boom"]).astype(np.int32)

    if derived_cols:
        snap = pd.concat([snap, pd.DataFrame(derived_cols, index=snap.index)], axis=1)

    g = snap.groupby(group_key, group_keys=False, sort=False)
    new_cols = {}
    slope_fn = _slope_apply_factory()

    rolling_targets = [
        "payment_ratio", "min_pay_ratio",
        "utilization", "balance", "credit_limit",
        "monthly_income", "balance_to_income", "limit_to_income",
        "external_score_delta", "external_score",
        "tx_count_30d", "avg_tx_amt_90d", "overlimit_rate_90d",
    ]
    rolling_targets = [c for c in rolling_targets if c in snap.columns]

    for col in rolling_targets:
        for k in WINDOWS:
            r = g[col].rolling(k, min_periods=1)
            new_cols[f"{col}_mean_{k}m"] = r.mean().reset_index(level=0, drop=True).astype(np.float32)
            new_cols[f"{col}_max_{k}m"]  = r.max().reset_index(level=0, drop=True).astype(np.float32)
            new_cols[f"{col}_min_{k}m"]  = r.min().reset_index(level=0, drop=True).astype(np.float32)

            r2 = g[col].rolling(k, min_periods=2)
            new_cols[f"{col}_std_{k}m"] = r2.std().reset_index(level=0, drop=True).fillna(0.0).astype(np.float32)
            new_cols[f"{col}_trend_{k}m"] = (
                r2.apply(slope_fn, raw=True)
                  .reset_index(level=0, drop=True)
                  .fillna(0.0)
                  .astype(np.float32)
            )

        mean6_key = f"{col}_mean_6m"
        if mean6_key in new_cols:
            new_cols[f"{col}_last_vs_mean_6m"] = (snap[col].astype(np.float32) - new_cols[mean6_key]).astype(np.float32)

    if "status_id" in snap.columns:
        for k in WINDOWS:
            new_cols[f"status_worst_{k}m"] = (
                g["status_id"].rolling(k, min_periods=1).max()
                .reset_index(level=0, drop=True).astype(np.int32)
            )

    if "dpd_flag" in snap.columns:
        for k in WINDOWS:
            new_cols[f"dpd_any_{k}m"] = (
                g["dpd_flag"].rolling(k, min_periods=1).max()
                .reset_index(level=0, drop=True).astype(np.int32)
            )
            r2 = g["dpd_flag"].rolling(k, min_periods=2)
            new_cols[f"dpd_trend_{k}m"] = (
                r2.apply(slope_fn, raw=True)
                  .reset_index(level=0, drop=True)
                  .fillna(0.0).astype(np.float32)
            )

    if "severe_dpd_flag" in snap.columns:
        for k in WINDOWS:
            new_cols[f"severe_dpd_any_{k}m"] = (
                g["severe_dpd_flag"].rolling(k, min_periods=1).max()
                .reset_index(level=0, drop=True).astype(np.int32)
            )

    if "is_cli" in snap.columns:
        new_cols["cli_count_12m"] = (
            g["is_cli"].rolling(12, min_periods=1).sum()
            .reset_index(level=0, drop=True).astype(np.float32)
        )
    if "is_cld" in snap.columns:
        new_cols["cld_count_12m"] = (
            g["is_cld"].rolling(12, min_periods=1).sum()
            .reset_index(level=0, drop=True).astype(np.float32)
        )

    if "action_id" in snap.columns:
        midx = g.cumcount().astype(np.int32)
        if "is_cli" in snap.columns:
            last_cli = midx.where(snap["is_cli"] == 1).groupby(snap[group_key]).ffill()
            new_cols["months_since_cli"] = (midx - last_cli).fillna(999).astype(np.int32)
        if "is_cld" in snap.columns:
            last_cld = midx.where(snap["is_cld"] == 1).groupby(snap[group_key]).ffill()
            new_cols["months_since_cld"] = (midx - last_cld).fillna(999).astype(np.int32)

    if "is_recession" in snap.columns:
        for k in WINDOWS:
            new_cols[f"recession_frac_{k}m"] = (
                g["is_recession"].rolling(k, min_periods=1).mean()
                .reset_index(level=0, drop=True).astype(np.float32)
            )
    if "is_boom" in snap.columns:
        for k in WINDOWS:
            new_cols[f"boom_frac_{k}m"] = (
                g["is_boom"].rolling(k, min_periods=1).mean()
                .reset_index(level=0, drop=True).astype(np.float32)
            )

    if new_cols:
        snap = pd.concat([snap, pd.DataFrame(new_cols, index=snap.index)], axis=1)

    snap = snap.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return snap.copy()


# -----------------------------
# Monthly-only driver
# -----------------------------
def list_months(in_dir: Path):
    months = []
    for p in in_dir.glob("snapshots_base_*.parquet"):
        m = p.stem.replace("snapshots_base_", "")
        if len(m) == 7 and m[4] == "-":
            months.append(m)
    return sorted(months)


def month_window(months, target, lookback):
    if target not in months:
        raise ValueError(f"Month not found: {target}")
    i = months.index(target)
    start = max(0, i - lookback)
    return months[start:i + 1]


def run_one_month(in_dir: Path, out_dir: Path, month: str, lookback: int):
    months = list_months(in_dir)
    window = month_window(months, month, lookback)

    dfs = []
    for m in window:
        dfs.append(pd.read_parquet(in_dir / f"snapshots_base_{m}.parquet", engine=PARQUET_ENGINE))
    hist = pd.concat(dfs, ignore_index=True)

    # Ensure date types (your base snapshots already contain statement_date)
    hist["statement_date"] = pd.to_datetime(hist["statement_date"])

    # Run SAME rolling logic on window
    hist_feat = add_rolling_stats(hist, group_key="cust_id", date_col="statement_date")

    # Slice current month only
    if "year_month" not in hist_feat.columns:
        # Safety: derive it if somehow missing
        hist_feat["year_month"] = pd.to_datetime(hist_feat["statement_date"]).dt.to_period("M").astype(str)

    cur = hist_feat[hist_feat["year_month"] == month].copy()

    out_path = out_dir / f"snapshots_{month}.parquet"
    write_parquet(cur, str(out_path))
    print(f"[FE] Wrote {out_path}")

def run_batch_monthly_outputs(in_dir: Path, out_dir: Path):
    """
    FAST initial build:
    - reads ALL snapshots_base_YYYY-MM.parquet
    - computes rolling features ONCE (like yearly/all_years style)
    - writes snapshots_YYYY-MM.parquet per month
    """
    months = list_months(in_dir)
    dfs = []
    for m in months:
        dfs.append(pd.read_parquet(in_dir / f"snapshots_base_{m}.parquet", engine=PARQUET_ENGINE))
    snap = pd.concat(dfs, ignore_index=True)

    snap["statement_date"] = pd.to_datetime(snap["statement_date"])
    feat = add_rolling_stats(snap, group_key="cust_id", date_col="statement_date")

    if "year_month" not in feat.columns:
        feat["year_month"] = pd.to_datetime(feat["statement_date"]).dt.to_period("M").astype(str)

    for ym, dfm in feat.groupby("year_month", sort=True):
        out_path = out_dir / f"snapshots_{ym}.parquet"
        write_parquet(dfm, str(out_path))
        print(f"[FE][BATCH] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing snapshots_base_YYYY-MM.parquet")
    ap.add_argument("--out_dir", required=True, help="Directory to write snapshots_YYYY-MM.parquet")
    ap.add_argument("--month", help="YYYY-MM (if not given, runs all months)")
    ap.add_argument("--mode", choices=["batch", "incremental"], default="batch",help="batch=compute rolling once and write monthly outputs; incremental=per-month lookback")

    ap.add_argument("--lookback_months", type=int, default=12)
    args = ap.parse_args()

    _require_pyarrow()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = list_months(in_dir)
    if not months:
        raise RuntimeError(f"No snapshots_base_YYYY-MM.parquet found in {in_dir}")

    if args.mode == "batch":
        run_batch_monthly_outputs(in_dir, out_dir)
    else:
        # incremental mode (per month)
        if args.month:
            run_one_month(in_dir, out_dir, args.month, args.lookback_months)
        else:
            for m in months:
                run_one_month(in_dir, out_dir, m, args.lookback_months)


    print("Done. Wrote feature-engineered snapshots per month:")
    print(f" - {out_dir}/snapshots_YYYY-MM.parquet")


if __name__ == "__main__":
    main()
