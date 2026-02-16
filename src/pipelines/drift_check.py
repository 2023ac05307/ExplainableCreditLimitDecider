#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.monitoring.push_metrics import push_job_status
import json
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-8) -> float:
    """
    PSI for numeric arrays. Uses quantile bins from expected.
    """
    expected = expected.astype(np.float64)
    actual = actual.astype(np.float64)

    # Handle constant expected
    if np.nanstd(expected) < 1e-12:
        return 0.0

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.nanquantile(expected, qs)
    cuts = np.unique(cuts)
    if len(cuts) <= 2:
        return 0.0

    exp_counts, _ = np.histogram(expected[~np.isnan(expected)], bins=cuts)
    act_counts, _ = np.histogram(actual[~np.isnan(actual)], bins=cuts)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1.0)
    act_perc = np.clip(act_perc, eps, 1.0)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))


@dataclass
class DriftCheckConfig:
    baseline_parquet: str                 # e.g., train_aug last-month rows OR train snapshots
    current_parquet: str                  # e.g., latest month last snapshots to score
    out_dir: str = "reports/drift"

    # only features used by models (recommended)
    feature_prefix: str = "s_"            # use s_ features
    ignore_features: Optional[List[str]] = None

    psi_bins: int = 10
    psi_warn: float = 0.10                # common PSI threshold
    psi_alert: float = 0.25
    alert_frac: float = 0.05              # drift if >= 5% features beyond psi_alert

    # optional filter
    max_rows_baseline: Optional[int] = 500_000
    max_rows_current: Optional[int] = 200_000


def run_drift_check(conf: DriftCheckConfig) -> Dict[str, Any]:
    out = Path(conf.out_dir)
    _ensure_dir(out)

    base = _read_parquet(conf.baseline_parquet)
    cur = _read_parquet(conf.current_parquet)

    if conf.max_rows_baseline and len(base) > conf.max_rows_baseline:
        base = base.sample(conf.max_rows_baseline, random_state=42)
    if conf.max_rows_current and len(cur) > conf.max_rows_current:
        cur = cur.sample(conf.max_rows_current, random_state=42)

    ignore = set(conf.ignore_features or [])
    feat_cols = [c for c in base.columns if c.startswith(conf.feature_prefix) and c in cur.columns and c not in ignore]
    if not feat_cols:
        raise RuntimeError(f"No matching features with prefix='{conf.feature_prefix}' found between baseline/current.")

    rows = []
    warn_cnt = 0
    alert_cnt = 0

    for c in feat_cols:
        b = pd.to_numeric(base[c], errors="coerce").to_numpy()
        a = pd.to_numeric(cur[c], errors="coerce").to_numpy()
        psi = _psi(b, a, bins=conf.psi_bins)

        level = "OK"
        if psi >= conf.psi_alert:
            level = "ALERT"
            alert_cnt += 1
        elif psi >= conf.psi_warn:
            level = "WARN"
            warn_cnt += 1

        rows.append({"feature": c, "psi": float(psi), "level": level})

    report = pd.DataFrame(rows).sort_values(["psi"], ascending=False)
    report_path = out / "drift_report.parquet"
    report.to_parquet(report_path, index=False)

    alert_frac = alert_cnt / max(len(feat_cols), 1)
    drift_detected = alert_frac >= conf.alert_frac

    summary = {
        "drift_detected": drift_detected,
        "n_features": len(feat_cols),
        "warn_cnt": warn_cnt,
        "alert_cnt": alert_cnt,
        "alert_frac": alert_frac,
        "psi_warn": conf.psi_warn,
        "psi_alert": conf.psi_alert,
        "alert_frac_threshold": conf.alert_frac,
        "baseline_rows": int(len(base)),
        "current_rows": int(len(cur)),
    }

    summary_path = out / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {"summary": summary, "report_parquet": str(report_path), "summary_json": str(summary_path)}


def main():
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--out_dir", default="reports/drift")
    args = p.parse_args()

    t0 = time.time()
    ok = False

    try:
        conf = DriftCheckConfig(
            baseline_parquet=args.baseline,
            current_parquet=args.current,
            out_dir=args.out_dir,
        )

        out = run_drift_check(conf)
        summary = out["summary"]

        print(json.dumps(summary, indent=2))

        ok = True

        # ---- BUSINESS METRICS TO PROMETHEUS ----
        push_job_status(
            job_name="drift_check",
            ok=not summary["drift_detected"],   # fail if drift detected
            duration_s=time.time() - t0,
            extra={
                "drift_detected": int(summary["drift_detected"]),
                "alert_feature_count": summary["alert_cnt"],
                "warn_feature_count": summary["warn_cnt"],
                "drift_alert_fraction": summary["alert_frac"],
                "features_checked": summary["n_features"],
            },
        )

    except Exception as e:
        # ---- FAILURE SIGNAL ----
        push_job_status(
            job_name="drift_check",
            ok=False,
            duration_s=time.time() - t0,
        )
        raise

