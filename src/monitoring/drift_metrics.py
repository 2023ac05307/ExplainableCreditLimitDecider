from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config + Result Types
# -----------------------------
@dataclass(frozen=True)
class DriftThresholds:
    psi_low: float = 0.10
    psi_medium: float = 0.20
    psi_high: float = 0.30


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    severity: str  # "low" | "medium" | "high" | "none"
    ref_count: int
    cur_count: int
    ref_missing_rate: float
    cur_missing_rate: float


@dataclass(frozen=True)
class DriftReport:
    reference_path: str
    current_path: str
    feature_prefix: str
    n_features: int
    thresholds: DriftThresholds
    overall_severity: str
    top_features: List[FeatureDrift]
    summary: Dict[str, int]  # counts by severity
    generated_at_utc: str


# -----------------------------
# Helpers
# -----------------------------
def _utc_now_iso() -> str:
    # No external deps; UTC timestamp string
    return pd.Timestamp.utcnow().isoformat()


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    # Coerce numeric while preserving NaN
    return pd.to_numeric(series, errors="coerce")


def _psi_from_hist(ref_counts: np.ndarray, cur_counts: np.ndarray, eps: float = 1e-12) -> float:
    """PSI(ref || cur) using counts (not normalized)."""
    ref = ref_counts.astype(np.float64)
    cur = cur_counts.astype(np.float64)

    ref_sum = ref.sum()
    cur_sum = cur.sum()
    if ref_sum <= 0 or cur_sum <= 0:
        return 0.0

    ref_pct = np.maximum(ref / ref_sum, eps)
    cur_pct = np.maximum(cur / cur_sum, eps)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    if not np.isfinite(psi):
        return 0.0
    return float(psi)


def _psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Numeric PSI via quantile bins on reference distribution."""
    ref = _safe_to_numeric(ref).dropna()
    cur = _safe_to_numeric(cur).dropna()

    if len(ref) < 50 or len(cur) < 50:
        return 0.0

    # If constant or nearly constant reference, PSI not meaningful
    if float(ref.nunique()) <= 1:
        return 0.0

    # Quantile bin edges from reference
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref.values, qs))
    if len(edges) <= 2:
        return 0.0

    ref_counts, _ = np.histogram(ref.values, bins=edges)
    cur_counts, _ = np.histogram(cur.values, bins=edges)
    return _psi_from_hist(ref_counts, cur_counts)


def _psi_categorical(ref: pd.Series, cur: pd.Series, top_k: int = 50) -> float:
    """Categorical PSI using top_k categories from reference + 'OTHER' bucket."""
    ref = ref.astype("object").fillna("__MISSING__")
    cur = cur.astype("object").fillna("__MISSING__")

    ref_vc = ref.value_counts(dropna=False)
    top = list(ref_vc.head(top_k).index)
    # Map non-top categories to OTHER
    ref_m = ref.where(ref.isin(top), other="__OTHER__")
    cur_m = cur.where(cur.isin(top), other="__OTHER__")

    cats = list(pd.Index(ref_m.unique()).union(pd.Index(cur_m.unique())))
    ref_counts = np.array([int((ref_m == c).sum()) for c in cats], dtype=np.int64)
    cur_counts = np.array([int((cur_m == c).sum()) for c in cats], dtype=np.int64)
    return _psi_from_hist(ref_counts, cur_counts)


def _severity(psi: float, thr: DriftThresholds) -> str:
    if psi >= thr.psi_high:
        return "high"
    if psi >= thr.psi_medium:
        return "medium"
    if psi >= thr.psi_low:
        return "low"
    return "none"


def _pick_feature_cols(df: pd.DataFrame, feature_prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(feature_prefix)]


def _read_parquet_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        # Directory of parquet parts
        parts = sorted([x for x in p.glob("*.parquet")])
        if not parts:
            raise FileNotFoundError(f"No parquet parts found in directory: {path}")
        return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(p)


# -----------------------------
# Public API
# -----------------------------
def compute_drift_report(
    reference_path: str,
    current_path: str,
    *,
    feature_prefix: str = "s_",
    thresholds: Optional[DriftThresholds] = None,
    numeric_bins: int = 10,
    cat_top_k: int = 50,
    max_features_in_report: int = 30,
    sample_ref: Optional[int] = 200_000,
    sample_cur: Optional[int] = 200_000,
    random_state: int = 42,
) -> DriftReport:
    """
    Compute drift report using PSI feature-wise.
    Reads parquet file OR parquet directory (partitioned output).
    """
    thr = thresholds or DriftThresholds()

    ref = _read_parquet_any(reference_path)
    cur = _read_parquet_any(current_path)

    feat_cols = _pick_feature_cols(ref, feature_prefix)
    if not feat_cols:
        raise ValueError(f"No feature columns found with prefix '{feature_prefix}' in reference dataset.")

    # Keep only shared columns to avoid schema drift breaking
    shared = [c for c in feat_cols if c in cur.columns]
    if not shared:
        raise ValueError("No shared feature columns between reference and current datasets.")

    ref = ref[shared]
    cur = cur[shared]

    # Sampling for speed
    if sample_ref and len(ref) > sample_ref:
        ref = ref.sample(n=sample_ref, random_state=random_state)
    if sample_cur and len(cur) > sample_cur:
        cur = cur.sample(n=sample_cur, random_state=random_state)

    out: List[FeatureDrift] = []

    for c in shared:
        ref_s = ref[c]
        cur_s = cur[c]

        ref_missing = float(ref_s.isna().mean())
        cur_missing = float(cur_s.isna().mean())

        # Decide type: numeric if can coerce many values; else categorical
        ref_num = _safe_to_numeric(ref_s)
        cur_num = _safe_to_numeric(cur_s)

        numeric_ratio = float(ref_num.notna().mean())  # how many look numeric
        if numeric_ratio >= 0.80:
            psi = _psi_numeric(ref_s, cur_s, bins=numeric_bins)
        else:
            psi = _psi_categorical(ref_s, cur_s, top_k=cat_top_k)

        sev = _severity(psi, thr)
        out.append(
            FeatureDrift(
                feature=c,
                psi=float(psi),
                severity=sev,
                ref_count=int(len(ref_s)),
                cur_count=int(len(cur_s)),
                ref_missing_rate=ref_missing,
                cur_missing_rate=cur_missing,
            )
        )

    # Summaries
    counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
    for r in out:
        counts[r.severity] = counts.get(r.severity, 0) + 1

    # Overall severity = max severity across features
    severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    overall = max(out, key=lambda x: severity_order[x.severity]).severity if out else "none"

    # Top features by PSI
    out_sorted = sorted(out, key=lambda x: x.psi, reverse=True)
    top = out_sorted[: max_features_in_report]

    return DriftReport(
        reference_path=reference_path,
        current_path=current_path,
        feature_prefix=feature_prefix,
        n_features=len(shared),
        thresholds=thr,
        overall_severity=overall,
        top_features=top,
        summary=counts,
        generated_at_utc=_utc_now_iso(),
    )


def save_drift_report(report: DriftReport, out_json_path: str) -> str:
    p = Path(out_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(report)
    # dataclass inside dataclass -> ensure thresholds + top_features render nicely
    payload["thresholds"] = asdict(report.thresholds)
    payload["top_features"] = [asdict(x) for x in report.top_features]

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute PSI-based drift report for s_* features.")
    ap.add_argument("--ref", required=True, help="Reference parquet file OR parquet directory")
    ap.add_argument("--cur", required=True, help="Current parquet file OR parquet directory")
    ap.add_argument("--out", default="reports/drift/drift_summary.json", help="Output JSON path")
    ap.add_argument("--feature-prefix", default="s_", help="Feature prefix to include")
    ap.add_argument("--bins", type=int, default=10, help="Quantile bins for numeric PSI")
    ap.add_argument("--topk", type=int, default=50, help="Top-K categories for categorical PSI")
    ap.add_argument("--max-features", type=int, default=30, help="Max features included in report")
    args = ap.parse_args()

    rep = compute_drift_report(
        reference_path=args.ref,
        current_path=args.cur,
        feature_prefix=args.feature_prefix,
        numeric_bins=args.bins,
        cat_top_k=args.topk,
        max_features_in_report=args.max_features,
    )
    path = save_drift_report(rep, args.out)
    print(f"[OK] Drift report written: {path}")
    print(f"Overall severity: {rep.overall_severity} | summary={rep.summary}")
