#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
augment_counterfactual_cli_cld_parquet.py
-----------------------------------------
Parquet-IN + Parquet-OUT (NO CSV support)

Creates counterfactual (CF) CLI/CLD examples from HOLD rows, but ensures CF rows are NOT
"almost HOLD" by:
  - tiered candidate selection (strong/medium/weak)
  - tiered sampling probs to meet targets
  - tiered sample_weight to avoid weak CF polluting training
  - optional magnitude variants per selected row (e.g., 10/20/30/40%)

Output is a Parquet *dataset directory* containing many part-*.parquet files.

Requires:
  pip install pyarrow

Example:
  python augment_counterfactual_cli_cld_parquet.py ^
    --in-traj rl_dataset/trajectories_strict.parquet ^
    --out-dir rl_dataset/trajectories_strict_aug_parquet ^
    --cli-mult 10 --cld-mult 3 ^
    --cli-variants 2 --cld-variants 2
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("This script requires pyarrow. Install with: pip install pyarrow") from e


ACTION_HOLD = 0
ACTION_CLI  = 1
ACTION_CLD  = 2

MAG_MAX = 0.40
EPS = 1e-6


# ------------------ utils ------------------

def exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns

def safe_float_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    if s is None:
        return pd.Series(default)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(np.float32)

def safe_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    if s is None:
        return pd.Series(default)
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)

def getf(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return safe_float_series(df.get(col, pd.Series(default, index=df.index)), default)

def geti(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    return safe_int_series(df.get(col, pd.Series(default, index=df.index)), default)

def compute_util(balance: pd.Series, limit: pd.Series) -> pd.Series:
    b = safe_float_series(balance, 0.0)
    l = safe_float_series(limit, 1.0)
    l = l.where(l > 0, 1.0)
    return (b / l).astype(np.float32)

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _adjust_rolling_mean(df: pd.DataFrame, base_prefix: str, old_last: np.ndarray, new_last: np.ndarray, k: int):
    """
    Approx update: mean_new = mean_old + (new_last - old_last)/k
    Only touches mean_{k}m and last_vs_mean_6m (if k==6).
    """
    mean_col = f"{base_prefix}_mean_{k}m"
    if not exists(df, mean_col):
        return
    mean_old = safe_float_series(df[mean_col], 0.0).to_numpy(np.float32)
    mean_new = mean_old + (new_last - old_last) / float(k)
    df[mean_col] = mean_new.astype(np.float32)

    if k == 6:
        lvm = f"{base_prefix}_last_vs_mean_6m"
        if exists(df, lvm):
            df[lvm] = (new_last - mean_new).astype(np.float32)


# ------------------ magnitude heuristics ------------------

def compute_mag_cli(util: float, dpd12: float, score: float) -> float:
    u = np.clip(util, 0.0, 1.2)
    d = np.clip(dpd12 / 6.0, 0.0, 1.0)
    sc = np.clip((score - 600.0) / 200.0, 0.0, 1.0)
    m = 0.10 + 0.25 * u + 0.10 * sc - 0.20 * d
    return float(np.clip(m, 0.05, MAG_MAX))

def compute_mag_cld(util: float, dpd12: float, score: float) -> float:
    u = np.clip(util, 0.0, 1.5)
    d = np.clip(dpd12 / 4.0, 0.0, 1.0)
    sc = np.clip((650.0 - score) / 250.0, 0.0, 1.0)
    m = 0.10 + 0.20 * np.clip(u - 0.6, 0.0, 1.0) + 0.25 * d + 0.10 * sc
    return float(np.clip(m, 0.05, MAG_MAX))


# ------------------ Candidate selection (tiers) ------------------

def pick_cli_candidates_tier(df: pd.DataFrame) -> pd.Series:
    a = geti(df, "action_id", 0)
    status_id = geti(df, "s_status_id", 0)

    dpd12  = getf(df, "s_dpd_count_12m", 0.0)
    score  = getf(df, "s_external_score", 700.0)
    payr   = getf(df, "s_payment_ratio", 1.0)
    tx30   = getf(df, "s_tx_count_30d", 0.0)

    util = compute_util(df.get("s_balance", pd.Series(0.0, index=df.index)),
                        df.get("s_credit_limit", pd.Series(1.0, index=df.index)))

    has_ts = ("s_payment_ratio_trend_6m" in df.columns)

    pay_tr6   = getf(df, "s_payment_ratio_trend_6m", 0.0)
    score_tr6 = getf(df, "s_external_score_delta_trend_6m", 0.0)
    tx_tr6    = getf(df, "s_tx_count_30d_trend_6m", 0.0)

    dpd_any_6 = getf(df, "s_dpd_any_6m", 0.0)
    sev_any_6 = getf(df, "s_severe_dpd_any_6m", 0.0)
    worst6    = getf(df, "s_status_worst_6m", status_id.astype(np.float32))

    rec6      = getf(df, "s_recession_frac_6m", 0.0)
    is_rec    = getf(df, "s_is_recession", 0.0)

    base = (a == ACTION_HOLD)

    demand_strong = ((util >= 0.45) | (tx30 >= 4) | (tx_tr6 > 0.0))
    demand_medium = ((util >= 0.35) | (tx30 >= 3) | (tx_tr6 > -0.2))
    demand_weak   = ((util >= 0.25) | (tx30 >= 2))

    macro_ok_strong = ((rec6 <= 0.50) & (is_rec <= 0.0))
    macro_ok_medium = ((rec6 <= 0.75))

    t1 = (
        (dpd12 <= 1.0) &
        (payr >= 0.85) &
        (score >= 670.0) &
        (status_id <= 2) &
        (dpd_any_6 <= 0.0) &
        (sev_any_6 <= 0.0) &
        (worst6 <= 1.0) &
        (pay_tr6 >= -0.01) &
        (score_tr6 >= -0.5) &
        demand_strong &
        macro_ok_strong
    )

    t2 = (
        (dpd12 <= 2.0) &
        (payr >= 0.75) &
        (score >= 640.0) &
        (status_id <= 3) &
        (sev_any_6 <= 0.0) &
        (worst6 <= 2.0) &
        (pay_tr6 >= -0.03) &
        (score_tr6 >= -1.0) &
        demand_medium &
        macro_ok_medium
    )

    t3 = (
        (payr >= 0.65) &
        (score >= 620.0) &
        (status_id <= 3) &
        (sev_any_6 <= 0.0) &
        (worst6 <= 2.0) &
        demand_weak
    )

    tier = pd.Series(0, index=df.index, dtype=int)
    if has_ts:
        tier.loc[base & t3] = 3
        tier.loc[base & t2] = 2
        tier.loc[base & t1] = 1
    else:
        t1s = (base & (dpd12 <= 1.0) & (payr >= 0.85) & (score >= 670.0) & (util >= 0.45))
        t2s = (base & (dpd12 <= 2.0) & (payr >= 0.75) & (score >= 640.0) & (util >= 0.35))
        tier.loc[t2s] = 2
        tier.loc[t1s] = 1

    return tier


def pick_cld_candidates_tier(df: pd.DataFrame) -> pd.Series:
    a = geti(df, "action_id", 0)
    status_id = geti(df, "s_status_id", 0)

    dpd12  = getf(df, "s_dpd_count_12m", 0.0)
    score  = getf(df, "s_external_score", 700.0)
    payr   = getf(df, "s_payment_ratio", 1.0)
    over   = getf(df, "s_overlimit_rate_90d", 0.0)

    util = compute_util(df.get("s_balance", pd.Series(0.0, index=df.index)),
                        df.get("s_credit_limit", pd.Series(1.0, index=df.index)))

    has_ts = ("s_payment_ratio_trend_6m" in df.columns)

    pay_tr6   = getf(df, "s_payment_ratio_trend_6m", 0.0)
    score_tr6 = getf(df, "s_external_score_delta_trend_6m", 0.0)
    over_tr6  = getf(df, "s_overlimit_rate_90d_trend_6m", 0.0)

    dpd_any_6 = getf(df, "s_dpd_any_6m", 0.0)
    sev_any_6 = getf(df, "s_severe_dpd_any_6m", 0.0)
    worst6    = getf(df, "s_status_worst_6m", status_id.astype(np.float32))

    rec6      = getf(df, "s_recession_frac_6m", 0.0)
    is_rec    = getf(df, "s_is_recession", 0.0)

    base = (a == ACTION_HOLD) & (status_id <= 3)

    t1 = (
        base &
        (
            (sev_any_6 >= 1.0) |
            (worst6 >= 2.0) |
            (dpd_any_6 >= 1.0) |
            (dpd12 >= 3.0) |
            (payr <= 0.60) |
            (pay_tr6 <= -0.03) |
            (score <= 640.0) |
            (score_tr6 <= -1.0) |
            (util >= 0.95) |
            (over >= 0.06) |
            (over_tr6 >= 0.015) |
            (is_rec >= 1.0) |
            (rec6 >= 0.50)
        )
    )

    t2 = (
        base &
        (
            (worst6 >= 2.0) |
            (dpd_any_6 >= 1.0) |
            (dpd12 >= 2.0) |
            (payr <= 0.68) |
            (pay_tr6 <= -0.02) |
            (score <= 655.0) |
            (score_tr6 <= -0.5) |
            (util >= 0.90) |
            (over >= 0.05) |
            (rec6 >= 0.40)
        )
    )

    t3 = (
        base &
        (
            ((util >= 0.85) & (payr <= 0.72)) |
            (dpd12 >= 2.0) |
            (over >= 0.05)
        )
    )

    tier = pd.Series(0, index=df.index, dtype=int)
    if has_ts:
        tier.loc[t3] = 3
        tier.loc[t2] = 2
        tier.loc[t1] = 1
    else:
        t1s = base & ((dpd12 >= 3.0) | (payr <= 0.60) | (util >= 0.95) | (score <= 640.0) | (over >= 0.06))
        t2s = base & ((dpd12 >= 2.0) | (payr <= 0.68) | (util >= 0.90) | (score <= 655.0) | (over >= 0.05))
        tier.loc[t2s] = 2
        tier.loc[t1s] = 1

    return tier


# ------------------ Counterfactual edit ------------------

def apply_counterfactual(df_chunk: pd.DataFrame,
                         action: int,
                         credit_floor: float,
                         credit_cap: float,
                         reward_cli_k: float,
                         reward_cld_k: float,
                         cf_weight: float,
                         cf_tag: str,
                         fixed_mag: float | None = None) -> pd.DataFrame:

    out = df_chunk.copy()

    s_lim  = getf(out, "s_credit_limit", 0.0)
    s1_lim_old = getf(out, "s1_credit_limit", 0.0)

    base_lim = s_lim.copy()
    base_lim[base_lim <= 0] = s1_lim_old[base_lim <= 0]
    base_lim[base_lim <= 0] = 20000.0

    util = compute_util(out.get("s_balance", pd.Series(0.0, index=out.index)), base_lim)
    s_dpd  = getf(out, "s_dpd_count_12m", 0.0)
    s_score = getf(out, "s_external_score", 700.0)

    if fixed_mag is not None:
        mags = np.full(len(out), float(np.clip(fixed_mag, 0.05, MAG_MAX)), dtype=np.float32)
    else:
        mags = np.zeros(len(out), dtype=np.float32)
        for i in range(len(out)):
            u = float(util.iloc[i])
            d = float(s_dpd.iloc[i])
            sc = float(s_score.iloc[i])
            mags[i] = compute_mag_cli(u, d, sc) if action == ACTION_CLI else compute_mag_cld(u, d, sc)
        mags = np.clip(mags, 0.0, MAG_MAX).astype(np.float32)

    if action == ACTION_CLI:
        new_lim = base_lim.to_numpy(np.float32) * (1.0 + mags)
        cf_action = "CLI"
    else:
        new_lim = base_lim.to_numpy(np.float32) * (1.0 - mags)
        cf_action = "CLD"

    new_lim = np.clip(new_lim, credit_floor, credit_cap).astype(np.float32)

    out["action_id"] = int(action)
    out["magnitude_pct"] = mags
    out["is_cf"] = 1
    out["sample_weight"] = float(cf_weight)
    out["cf_action"] = cf_action
    out["cf_rule_tag"] = cf_tag

    if exists(out, "s1_credit_limit"):
        out["s1_credit_limit"] = new_lim

    if exists(out, "s1_balance"):
        bal1 = getf(out, "s1_balance", 0.0).to_numpy(np.float32)
        if action == ACTION_CLI:
            bump = 0.015 * mags * _clip01(util.to_numpy(np.float32)) * new_lim
            out["s1_balance"] = (bal1 + bump).astype(np.float32)
        else:
            cut = 0.010 * mags * _clip01(util.to_numpy(np.float32)) * base_lim.to_numpy(np.float32)
            out["s1_balance"] = np.maximum(0.0, bal1 - cut).astype(np.float32)

    if exists(out, "s1_monthly_income"):
        inc1 = np.maximum(getf(out, "s1_monthly_income", 1.0).to_numpy(np.float32), 1.0)
        if exists(out, "s1_limit_to_income"):
            out["s1_limit_to_income"] = (new_lim / inc1).astype(np.float32)
        if exists(out, "s1_balance") and exists(out, "s1_balance_to_income"):
            b1 = np.maximum(getf(out, "s1_balance", 0.0).to_numpy(np.float32), 0.0)
            out["s1_balance_to_income"] = (b1 / inc1).astype(np.float32)

        if exists(out, "s1_credit_limit_mean_3m"):
            _adjust_rolling_mean(out, "s1_credit_limit", s1_lim_old.to_numpy(np.float32), new_lim, 3)
            _adjust_rolling_mean(out, "s1_credit_limit", s1_lim_old.to_numpy(np.float32), new_lim, 6)
            _adjust_rolling_mean(out, "s1_credit_limit", s1_lim_old.to_numpy(np.float32), new_lim, 12)

        if exists(out, "s1_limit_to_income_mean_3m") and exists(out, "s1_limit_to_income"):
            old_lti = getf(out, "s1_limit_to_income", 0.0).to_numpy(np.float32)
            new_lti = getf(out, "s1_limit_to_income", 0.0).to_numpy(np.float32)
            _adjust_rolling_mean(out, "s1_limit_to_income", old_lti, new_lti, 3)
            _adjust_rolling_mean(out, "s1_limit_to_income", old_lti, new_lti, 6)
            _adjust_rolling_mean(out, "s1_limit_to_income", old_lti, new_lti, 12)

    if exists(out, "s1_months_since_cli") and exists(out, "s1_months_since_cld"):
        m_cli = geti(out, "s1_months_since_cli", 999).to_numpy()
        m_cld = geti(out, "s1_months_since_cld", 999).to_numpy()
        if action == ACTION_CLI:
            out["s1_months_since_cli"] = np.zeros(len(out), dtype=int)
            out["s1_months_since_cld"] = np.minimum(m_cld + 1, 999).astype(int)
        else:
            out["s1_months_since_cld"] = np.zeros(len(out), dtype=int)
            out["s1_months_since_cli"] = np.minimum(m_cli + 1, 999).astype(int)

    if exists(out, "s1_cli_count_12m") and exists(out, "s1_cld_count_12m"):
        c_cli = geti(out, "s1_cli_count_12m", 0).to_numpy()
        c_cld = geti(out, "s1_cld_count_12m", 0).to_numpy()
        if action == ACTION_CLI:
            out["s1_cli_count_12m"] = np.minimum(c_cli + 1, 12).astype(int)
            out["s1_cld_count_12m"] = c_cld.astype(int)
        else:
            out["s1_cld_count_12m"] = np.minimum(c_cld + 1, 12).astype(int)
            out["s1_cli_count_12m"] = c_cli.astype(int)

    r = getf(out, "reward", 0.0).to_numpy(np.float32)
    dpd_n = _clip01((s_dpd.to_numpy(np.float32) / 12.0))
    util_n = np.clip(util.to_numpy(np.float32), 0.0, 1.5)
    score_n = _clip01((s_score.to_numpy(np.float32) - 300.0) / 600.0)

    if action == ACTION_CLI:
        delta_r = reward_cli_k * mags * (0.25 + 0.75 * _clip01(util_n)) * (0.5 + 0.5 * score_n) * (1.0 - 0.7 * dpd_n)
        out["reward"] = (r + delta_r).astype(np.float32)
    else:
        over = getf(out, "s_overlimit_rate_90d", 0.0).to_numpy(np.float32)
        risk = (0.6 * dpd_n + 0.4 * np.clip(util_n - 0.7, 0.0, 1.0) + 0.2 * np.clip(over, 0.0, 1.0))
        delta_r = reward_cld_k * mags * (risk - 0.12)
        out["reward"] = (r + delta_r).astype(np.float32)

    return out


# ------------------ tier planning ------------------

def plan_tier_probs(add_target: int, cand_t1: int, cand_t2: int, cand_t3: int):
    need = int(add_target)
    take1 = min(need, cand_t1); need -= take1
    take2 = min(need, cand_t2); need -= take2
    take3 = min(need, cand_t3); need -= take3

    p1 = 1.0 if cand_t1 > 0 else 0.0
    p2 = (take2 / cand_t2) if cand_t2 > 0 else 0.0
    p3 = (take3 / cand_t3) if cand_t3 > 0 else 0.0

    return float(p1), float(np.clip(p2, 0.0, 1.0)), float(np.clip(p3, 0.0, 1.0)), int(need)


# ------------------ Parquet IO helpers ------------------

def iter_parquet_rowgroups(path: str):
    """
    Yield pandas DataFrames for each row-group from a single Parquet file.
    This is the Parquet equivalent of CSV chunking.
    """
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        yield table.to_pandas()


def ensure_dir_empty(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # remove old parts
    for f in glob.glob(os.path.join(out_dir, "part-*.parquet")):
        os.remove(f)
    for f in glob.glob(os.path.join(out_dir, "_summary.json")):
        os.remove(f)


def write_part(df: pd.DataFrame, out_dir: str, part_idx: int, compression: str = "snappy"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    out_path = os.path.join(out_dir, f"part-{part_idx:06d}.parquet")
    pq.write_table(table, out_path, compression=compression)
    return out_path


def run_augment_counterfactual(
    in_traj: str,
    out_dir: str,
    rowgroup_limit: int = 0,
    cli_mult: float = 10.0,
    cld_mult: float = 3.0,
    credit_floor: float = 5000.0,
    credit_cap: float = 800000.0,
    reward_cli_k: float = 40.0,
    reward_cld_k: float = 35.0,
    cli_w1: float = 0.25,
    cli_w2: float = 0.15,
    cli_w3: float = 0.08,
    cld_w1: float = 0.25,
    cld_w2: float = 0.15,
    cld_w3: float = 0.08,
    cli_variants: int = 1,
    cld_variants: int = 1,
    seed: int = 123,
    compression: str = "snappy",
):
    """
    Import-friendly entry point for counterfactual augmentation.

    Input:
      - in_traj: single .parquet file (trajectories)
    Output:
      - out_dir: parquet dataset directory containing part-*.parquet + _summary.json

    Returns:
      - out_dir (str)
    """
    in_path = str(in_traj)
    out_dir = str(out_dir)

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)
    if not in_path.lower().endswith(".parquet"):
        raise ValueError("in_traj must be a single .parquet file (dataset dir not supported by this augmenter).")

    ensure_dir_empty(out_dir)

    buckets = [0.10, 0.20, 0.30, 0.40]
    cli_buckets = buckets[:max(1, min(4, int(cli_variants)))]
    cld_buckets = buckets[:max(1, min(4, int(cld_variants)))]

    rng = np.random.default_rng(int(seed))

    # ---------------- PASS 1 ----------------
    orig_counts = {ACTION_HOLD: 0, ACTION_CLI: 0, ACTION_CLD: 0}
    cli_t1 = cli_t2 = cli_t3 = 0
    cld_t1 = cld_t2 = cld_t3 = 0

    pf = pq.ParquetFile(in_path)
    for rg in range(pf.num_row_groups):
        if rowgroup_limit and rg >= rowgroup_limit:
            break
        chunk = pf.read_row_group(rg).to_pandas().replace([np.inf, -np.inf], np.nan)

        a = geti(chunk, "action_id", 0)
        orig_counts[ACTION_HOLD] += int((a == ACTION_HOLD).sum())
        orig_counts[ACTION_CLI]  += int((a == ACTION_CLI).sum())
        orig_counts[ACTION_CLD]  += int((a == ACTION_CLD).sum())

        t_cli = pick_cli_candidates_tier(chunk)
        cli_t1 += int((t_cli == 1).sum())
        cli_t2 += int((t_cli == 2).sum())
        cli_t3 += int((t_cli == 3).sum())

        t_cld = pick_cld_candidates_tier(chunk)
        cld_t1 += int((t_cld == 1).sum())
        cld_t2 += int((t_cld == 2).sum())
        cld_t3 += int((t_cld == 3).sum())

    add_cli_target = max(0, int(round(orig_counts[ACTION_CLI] * (cli_mult - 1.0))))
    add_cld_target = max(0, int(round(orig_counts[ACTION_CLD] * (cld_mult - 1.0))))

    p_cli1, p_cli2, p_cli3, unfilled_cli = plan_tier_probs(add_cli_target, cli_t1, cli_t2, cli_t3)
    p_cld1, p_cld2, p_cld3, unfilled_cld = plan_tier_probs(add_cld_target, cld_t1, cld_t2, cld_t3)

    # ---------------- PASS 2 ----------------
    total_out = 0
    added_cli = 0
    added_cld = 0
    wrote_parts = 0

    pf2 = pq.ParquetFile(in_path)

    def _sample_idx(idx_arr: np.ndarray, p: float) -> np.ndarray:
        if len(idx_arr) == 0 or p <= 0:
            return np.array([], dtype=idx_arr.dtype)
        return idx_arr[(rng.random(len(idx_arr)) < p)]

    for rg in range(pf2.num_row_groups):
        if rowgroup_limit and rg >= rowgroup_limit:
            break

        chunk = pf2.read_row_group(rg).to_pandas().replace([np.inf, -np.inf], np.nan)

        base = chunk.copy()
        base["is_cf"] = 0
        base["cf_action"] = ""
        base["cf_rule_tag"] = ""
        base["sample_weight"] = 1.0

        frames = [base]

        # ---- CLI CF ----
        t_cli = pick_cli_candidates_tier(chunk)
        idx_cli_t1 = chunk.index[(t_cli == 1)].to_numpy()
        idx_cli_t2 = chunk.index[(t_cli == 2)].to_numpy()
        idx_cli_t3 = chunk.index[(t_cli == 3)].to_numpy()

        for tier, p, w, idx_arr in [
            (1, p_cli1, cli_w1, idx_cli_t1),
            (2, p_cli2, cli_w2, idx_cli_t2),
            (3, p_cli3, cli_w3, idx_cli_t3),
        ]:
            sel = _sample_idx(idx_arr, p)
            if len(sel) == 0:
                continue
            for j, mag in enumerate(cli_buckets):
                tag = f"CLI_T{tier}_V{j+1}_M{int(mag*100)}"
                cf = apply_counterfactual(
                    chunk.loc[sel],
                    action=ACTION_CLI,
                    credit_floor=credit_floor,
                    credit_cap=credit_cap,
                    reward_cli_k=reward_cli_k,
                    reward_cld_k=reward_cld_k,
                    cf_weight=w,
                    cf_tag=tag,
                    fixed_mag=mag if j > 0 else None,
                )
                frames.append(cf)
                added_cli += len(cf)

        # ---- CLD CF ----
        t_cld = pick_cld_candidates_tier(chunk)
        idx_cld_t1 = chunk.index[(t_cld == 1)].to_numpy()
        idx_cld_t2 = chunk.index[(t_cld == 2)].to_numpy()
        idx_cld_t3 = chunk.index[(t_cld == 3)].to_numpy()

        for tier, p, w, idx_arr in [
            (1, p_cld1, cld_w1, idx_cld_t1),
            (2, p_cld2, cld_w2, idx_cld_t2),
            (3, p_cld3, cld_w3, idx_cld_t3),
        ]:
            sel = _sample_idx(idx_arr, p)
            if len(sel) == 0:
                continue
            for j, mag in enumerate(cld_buckets):
                tag = f"CLD_T{tier}_V{j+1}_M{int(mag*100)}"
                cf = apply_counterfactual(
                    chunk.loc[sel],
                    action=ACTION_CLD,
                    credit_floor=credit_floor,
                    credit_cap=credit_cap,
                    reward_cli_k=reward_cli_k,
                    reward_cld_k=reward_cld_k,
                    cf_weight=w,
                    cf_tag=tag,
                    fixed_mag=mag if j > 0 else None,
                )
                frames.append(cf)
                added_cld += len(cf)

        out = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
        write_part(out, out_dir, wrote_parts, compression=compression)
        wrote_parts += 1
        total_out += len(out)

    summary = {
        "input_parquet": in_path,
        "output_dir": out_dir,
        "parts_written": int(wrote_parts),
        "total_written_rows": int(total_out),
        "added_cli_cf": int(added_cli),
        "added_cld_cf": int(added_cld),
        "cli_mult": float(cli_mult),
        "cld_mult": float(cld_mult),
        "cli_variants": int(cli_variants),
        "cld_variants": int(cld_variants),
        "seed": int(seed),
        "compression": compression,
    }
    with open(os.path.join(out_dir, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_dir




# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-traj", required=True, help="Input trajectory parquet file (single .parquet)")
    ap.add_argument("--out-dir", required=True, help="Output directory for parquet dataset parts")

    # Parquet read is rowgroup-based; keep arg name for similarity
    ap.add_argument("--rowgroup-limit", type=int, default=0, help="0=all rowgroups, else stop after N rowgroups (demo)")

    ap.add_argument("--cli-mult", type=float, default=10.0)
    ap.add_argument("--cld-mult", type=float, default=3.0)

    ap.add_argument("--credit-floor", type=float, default=5000.0)
    ap.add_argument("--credit-cap", type=float, default=800000.0)

    ap.add_argument("--reward-cli-k", type=float, default=40.0)
    ap.add_argument("--reward-cld-k", type=float, default=35.0)

    ap.add_argument("--cli-w1", type=float, default=0.25)
    ap.add_argument("--cli-w2", type=float, default=0.15)
    ap.add_argument("--cli-w3", type=float, default=0.08)

    ap.add_argument("--cld-w1", type=float, default=0.25)
    ap.add_argument("--cld-w2", type=float, default=0.15)
    ap.add_argument("--cld-w3", type=float, default=0.08)

    ap.add_argument("--cli-variants", type=int, default=1, help="1..4 variants per selected CLI row")
    ap.add_argument("--cld-variants", type=int, default=1, help="1..4 variants per selected CLD row")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--compression", default="snappy", choices=["snappy", "zstd", "gzip", "brotli", "lz4_raw"])

    args = ap.parse_args()

    run_augment_counterfactual(
        in_traj=args.in_traj,
        out_dir=args.out_dir,
        rowgroup_limit=args.rowgroup_limit,
        cli_mult=args.cli_mult,
        cld_mult=args.cld_mult,
        credit_floor=args.credit_floor,
        credit_cap=args.credit_cap,
        reward_cli_k=args.reward_cli_k,
        reward_cld_k=args.reward_cld_k,
        cli_w1=args.cli_w1,
        cli_w2=args.cli_w2,
        cli_w3=args.cli_w3,
        cld_w1=args.cld_w1,
        cld_w2=args.cld_w2,
        cld_w3=args.cld_w3,
        cli_variants=args.cli_variants,
        cld_variants=args.cld_variants,
        seed=args.seed,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
