#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_trajectories_monthly_only.py
---------------------------------
INPUT : <snap_dir>/snapshots_YYYY-MM.parquet
OUTPUT:
  - <out_dir>/trajectories_YYYY-MM.parquet
  - <out_dir>/trajectories_strict_YYYY-MM.parquet

Monthly-only. Uses the SAME state_cols + shift(-1) trajectory construction
style as your original all_years script, but applied per month by stitching
(current_month + next_month) and then emitting only month=t transitions.

Requires:
  pip install pyarrow
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

WINDOWS = (3, 6, 12)
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "snappy"


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Parquet output requires 'pyarrow'. Install it with: pip install pyarrow"
        ) from e


def write_parquet(df: pd.DataFrame, path: str):
    _require_pyarrow()
    df.to_parquet(path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)


def list_months(snap_dir: Path):
    months = []
    for p in snap_dir.glob("snapshots_*.parquet"):
        m = p.stem.replace("snapshots_", "")
        if len(m) == 7 and m[4] == "-":
            months.append(m)
    return sorted(months)


def _state_cols_from_snap(snap: pd.DataFrame):
    base_state_cols = [
        "balance", "credit_limit", "utilization", "payment_ratio",
        "min_pay_ratio", "min_due", "status_id",
        "avg_tx_amt_90d", "tx_count_30d", "overlimit_rate_90d",
        "max_utilization_6m", "dpd_count_12m", "recent_cli_effectiveness",
        "external_score", "external_score_delta",
        "monthly_income", "balance_to_income", "limit_to_income",
        "macro_regime_id",
        "months_since_cli", "months_since_cld",
        "cli_count_12m", "cld_count_12m",
        "dpd_flag", "severe_dpd_flag",
    ]

    rolling_cols = [c for c in snap.columns if any(c.endswith(f"_{k}m") for k in WINDOWS)]
    extra_cols = [c for c in ["is_recession", "is_boom"] if c in snap.columns]

    state_cols = list(dict.fromkeys(base_state_cols + rolling_cols + extra_cols))
    state_cols = [c for c in state_cols if c in snap.columns]
    return state_cols


def build_month_trajectories(cur: pd.DataFrame, nxt: pd.DataFrame):
    """
    Build trajectories for month=t using (cur + nxt) stitched data, applying
    the same shift(-1) approach, but only keeping rows where t_date is in cur month.
    """
    # stitch and sort
    snap = pd.concat([cur, nxt], ignore_index=True)
    snap["statement_date"] = pd.to_datetime(snap["statement_date"])
    snap = snap.sort_values(["cust_id", "statement_date"]).reset_index(drop=True)

    # compute state columns
    state_cols = _state_cols_from_snap(snap)
    snap[state_cols] = snap[state_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    snap_next = snap.groupby("cust_id", sort=False).shift(-1)

    traj = pd.DataFrame(
        {
            "cust_id": snap["cust_id"].to_numpy(),
            "t_date": snap["statement_date"].to_numpy(),
            "t1_date": snap_next["statement_date"].to_numpy(),
            "action_id": pd.to_numeric(snap.get("action_id", 0), errors="coerce").fillna(0).astype(np.int32).to_numpy(),
            "magnitude_pct": pd.to_numeric(snap.get("magnitude_pct", 0.0), errors="coerce").fillna(0.0).astype(np.float32).to_numpy(),
            "reward": pd.to_numeric(snap.get("reward", 0.0), errors="coerce").fillna(0.0).astype(np.float32).to_numpy(),
        }
    )

    # SAME done logic as original:
    done = snap_next["statement_date"].isna()
    if "status_raw" in snap.columns:
        done = done | snap["status_raw"].isin(["default", "closed"])
    traj["done"] = done.astype(np.int8).to_numpy()

    s_df = snap[state_cols].copy()
    s_df.columns = [f"s_{c}" for c in s_df.columns]

    s1_df = snap_next[state_cols].copy()
    s1_df.columns = [f"s1_{c}" for c in s1_df.columns]

    traj = pd.concat([traj, s_df, s1_df], axis=1).copy()

    return traj

def append_strict_to_combined(traj_dir: Path, out_dir: Path):
    """
    Appends all trajectories_strict_YYYY-MM.parquet into a single
    trajectories_strict.parquet (training-ready dataset).

    Safe to re-run: always rebuilds deterministically.
    """
    strict_files = sorted(traj_dir.glob("trajectories_strict_*.parquet"))
    if not strict_files:
        print("[TRJ] No strict monthly files found, skipping combine step.")
        return

    dfs = []
    for p in strict_files:
        dfs.append(pd.read_parquet(p, engine=PARQUET_ENGINE))

    combined = pd.concat(dfs, ignore_index=True)

    # Deterministic ordering (important for reproducibility)
    combined = combined.sort_values(["cust_id", "t_date"]).reset_index(drop=True)

    out_path = out_dir / "trajectories_strict.parquet"
    combined.to_parquet(
        out_path,
        index=False,
        engine=PARQUET_ENGINE,
        compression=PARQUET_COMPRESSION,
    )

    print(f"[TRJ] Updated combined strict trajectories → {out_path}")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snap_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--month")
    p.add_argument("--combine_strict", action="store_true")
    args = p.parse_args()

    _require_pyarrow()

    snap_dir = Path(args.snap_dir)
    out_dir = Path(args.out_dir) if args.out_dir else snap_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    months = list_months(snap_dir)
    if not months:
        raise RuntimeError(f"No snapshots found in {snap_dir}")

    # ---- define run_month BEFORE using it ----
    def run_month(m):
        i = months.index(m)
        if i + 1 >= len(months):
            print(f"[TRJ] Skipping {m} (no next month available)")
            return

        nm = months[i + 1]
        cur = pd.read_parquet(snap_dir / f"snapshots_{m}.parquet", engine=PARQUET_ENGINE)
        nxt = pd.read_parquet(snap_dir / f"snapshots_{nm}.parquet", engine=PARQUET_ENGINE)

        traj_full = build_month_trajectories(cur, nxt)
        t_month = pd.to_datetime(traj_full["t_date"]).dt.to_period("M").astype(str)
        traj = traj_full[t_month == m].copy()
        traj_strict = traj[traj["t1_date"].notna()].copy()

        write_parquet(traj, out_dir / f"trajectories_{m}.parquet")
        write_parquet(traj_strict, out_dir / f"trajectories_strict_{m}.parquet")

        print(f"[TRJ] Wrote trajectories for {m}")

    # ---- ACTUAL execution ----
    if args.month:
        run_month(args.month)
    else:
        for m in months[:-1]:
            run_month(m)

    if args.combine_strict:
        append_strict_to_combined(out_dir, out_dir)

    print("[TRJ] Done")

if __name__ == "__main__":
    main()