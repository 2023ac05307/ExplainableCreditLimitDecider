from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SanityThresholds:
    max_missing_rate_per_feature: float = 0.50
    max_overall_missing_rate: float = 0.20
    min_rows: int = 10_000

    # Finance-safe ranges (adjust as needed)
    min_credit_limit: float = 0.0
    max_credit_limit: float = 2_000_000.0
    min_income: float = 0.0
    max_income: float = 50_000_000.0

    # Action distribution sanity (for 3-class trajectories)
    min_nonhold_rate: float = 0.005  # at least 0.5% non-hold
    max_nonhold_rate: float = 0.40   # avoid unrealistic oversampling

    # Duplicate key check (if keys exist)
    enforce_unique_key: bool = True


@dataclass(frozen=True)
class SanityReport:
    dataset_path: str
    n_rows: int
    n_cols: int
    passed: bool
    failures: List[str]
    warnings: List[str]
    stats: Dict[str, object]


def _read_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        parts = sorted([x for x in p.glob("*.parquet")])
        if not parts:
            raise FileNotFoundError(f"No parquet parts found in directory: {path}")
        return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(p)


def _col_exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def run_sanity_checks(
    dataset_path: str,
    *,
    thresholds: Optional[SanityThresholds] = None,
    feature_prefix: str = "s_",
    id_col: str = "cust_id",
    time_col: str = "month",
    action_col_candidates: Tuple[str, ...] = ("action_id", "action", "action_taken"),
    credit_limit_candidates: Tuple[str, ...] = ("s_credit_limit", "credit_limit"),
    income_candidates: Tuple[str, ...] = ("s_income", "income"),
) -> SanityReport:
    thr = thresholds or SanityThresholds()
    df = _read_parquet_any(dataset_path)

    failures: List[str] = []
    warnings: List[str] = []

    n_rows, n_cols = int(len(df)), int(df.shape[1])

    stats: Dict[str, object] = {
        "feature_prefix": feature_prefix,
        "thresholds": asdict(thr),
    }

    # ---- basic row count
    if n_rows < thr.min_rows:
        failures.append(f"Too few rows: {n_rows} < min_rows={thr.min_rows}")

    # ---- feature set
    feat_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if not feat_cols:
        failures.append(f"No feature columns found with prefix '{feature_prefix}'")
    stats["n_feature_cols"] = len(feat_cols)

    # ---- overall missingness
    overall_missing = float(df[feat_cols].isna().mean().mean()) if feat_cols else 1.0
    stats["overall_feature_missing_rate"] = overall_missing
    if overall_missing > thr.max_overall_missing_rate:
        failures.append(
            f"Overall feature missing rate too high: {overall_missing:.3f} > {thr.max_overall_missing_rate:.3f}"
        )

    # ---- per-feature missingness
    if feat_cols:
        miss = df[feat_cols].isna().mean().sort_values(ascending=False)
        top_miss = miss.head(20).to_dict()
        stats["top_missing_features"] = {k: float(v) for k, v in top_miss.items()}

        too_missing = miss[miss > thr.max_missing_rate_per_feature]
        if len(too_missing) > 0:
            failures.append(
                f"{len(too_missing)} features exceed max_missing_rate_per_feature={thr.max_missing_rate_per_feature:.2f}"
            )

    # ---- duplicate keys (if columns exist)
    if thr.enforce_unique_key and _col_exists(df, id_col) and _col_exists(df, time_col):
        dup = df.duplicated(subset=[id_col, time_col]).sum()
        stats["duplicate_keys"] = int(dup)
        if dup > 0:
            failures.append(f"Found duplicate rows for key ({id_col},{time_col}): {int(dup)}")
    else:
        stats["duplicate_keys"] = None

    # ---- numeric range checks: credit limit / income (if present)
    def _pick_first_existing(cands: Tuple[str, ...]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    cl_col = _pick_first_existing(credit_limit_candidates)
    if cl_col:
        cl = pd.to_numeric(df[cl_col], errors="coerce")
        stats["credit_limit_col"] = cl_col
        stats["credit_limit_min"] = float(np.nanmin(cl.values)) if np.isfinite(np.nanmin(cl.values)) else None
        stats["credit_limit_max"] = float(np.nanmax(cl.values)) if np.isfinite(np.nanmax(cl.values)) else None
        if (cl < thr.min_credit_limit).sum() > 0:
            failures.append(f"Credit limit has values < {thr.min_credit_limit} in column {cl_col}")
        if (cl > thr.max_credit_limit).sum() > 0:
            failures.append(f"Credit limit has values > {thr.max_credit_limit} in column {cl_col}")
    else:
        warnings.append("No credit limit column found (checked: s_credit_limit/credit_limit).")

    inc_col = _pick_first_existing(income_candidates)
    if inc_col:
        inc = pd.to_numeric(df[inc_col], errors="coerce")
        stats["income_col"] = inc_col
        stats["income_min"] = float(np.nanmin(inc.values)) if np.isfinite(np.nanmin(inc.values)) else None
        stats["income_max"] = float(np.nanmax(inc.values)) if np.isfinite(np.nanmax(inc.values)) else None
        if (inc < thr.min_income).sum() > 0:
            failures.append(f"Income has values < {thr.min_income} in column {inc_col}")
        if (inc > thr.max_income).sum() > 0:
            failures.append(f"Income has values > {thr.max_income} in column {inc_col}")
    else:
        warnings.append("No income column found (checked: s_income/income).")

    # ---- action distribution sanity (if present)
    action_col = _pick_first_existing(action_col_candidates)
    if action_col:
        stats["action_col"] = action_col
        a = df[action_col]

        # handle string actions OR numeric ids
        # nonhold definition: CLI/CLD OR ids 1/2
        if a.dtype.kind in ("i", "u", "f"):
            a_num = pd.to_numeric(a, errors="coerce").fillna(-1).astype(int)
            nonhold_rate = float((a_num.isin([1, 2])).mean())
            dist = a_num.value_counts(normalize=True).to_dict()
            stats["action_dist"] = {str(k): float(v) for k, v in dist.items()}
        else:
            a_str = a.astype(str).str.upper()
            nonhold_rate = float(a_str.isin(["CLI", "CLD"]).mean())
            dist = a_str.value_counts(normalize=True).to_dict()
            stats["action_dist"] = {str(k): float(v) for k, v in dist.items()}

        stats["nonhold_rate"] = nonhold_rate
        if nonhold_rate < thr.min_nonhold_rate:
            failures.append(f"Non-hold rate too low: {nonhold_rate:.4f} < {thr.min_nonhold_rate:.4f}")
        if nonhold_rate > thr.max_nonhold_rate:
            warnings.append(f"Non-hold rate very high: {nonhold_rate:.4f} > {thr.max_nonhold_rate:.4f} (oversampled?)")
    else:
        warnings.append("No action column found to validate action distribution.")

    passed = len(failures) == 0
    return SanityReport(
        dataset_path=dataset_path,
        n_rows=n_rows,
        n_cols=n_cols,
        passed=passed,
        failures=failures,
        warnings=warnings,
        stats=stats,
    )


def save_sanity_report(report: SanityReport, out_json_path: str) -> str:
    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return str(p)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Run production sanity checks on a dataset parquet.")
    ap.add_argument("--data", required=True, help="Parquet file OR parquet directory")
    ap.add_argument("--out", default="reports/sanity/sanity_report.json", help="Output JSON path")
    ap.add_argument("--feature-prefix", default="s_", help="Feature prefix to validate")
    ap.add_argument("--min-rows", type=int, default=10_000)
    ap.add_argument("--max-overall-missing", type=float, default=0.20)
    ap.add_argument("--max-per-feature-missing", type=float, default=0.50)
    args = ap.parse_args()

    thr = SanityThresholds(
        min_rows=args.min_rows,
        max_overall_missing_rate=args.max_overall_missing,
        max_missing_rate_per_feature=args.max_per_feature_missing,
    )

    rep = run_sanity_checks(args.data, thresholds=thr, feature_prefix=args.feature_prefix)
    path = save_sanity_report(rep, args.out)

    print(f"[OK] Sanity report written: {path}")
    print(f"PASSED={rep.passed}")
    if rep.failures:
        print("FAILURES:")
        for f in rep.failures:
            print(f" - {f}")
    if rep.warnings:
        print("WARNINGS:")
        for w in rep.warnings:
            print(f" - {w}")

    sys.exit(0 if rep.passed else 2)
