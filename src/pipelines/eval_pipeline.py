from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import json
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _quantiles(x: np.ndarray, qs=(0.90, 0.95)) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def _confusion_matrix_3cls(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t <= 2 and 0 <= p <= 2:
            cm[int(t), int(p)] += 1
    return cm


# ---------------------------
# NEW: Probability & drift helpers (safe additions)
# ---------------------------

def _sigmoid_clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p.astype(float), eps, 1.0 - eps)


def _logloss(y_true01: np.ndarray, p1: np.ndarray) -> float:
    p = _sigmoid_clip(p1)
    y = y_true01.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y_true01: np.ndarray, p1: np.ndarray) -> float:
    p = _sigmoid_clip(p1)
    y = y_true01.astype(float)
    return float(np.mean((p - y) ** 2))


def _ece(y_true01: np.ndarray, p1: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins in [0,1].
    """
    p = _sigmoid_clip(p1)
    y = y_true01.astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        w = float(np.mean(m))
        ece += w * abs(acc - conf)

    return float(ece)


def _safe_auc(y_true01: np.ndarray, p1: np.ndarray) -> float:
    """
    ROC-AUC if sklearn is available; else NaN.
    """
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true01)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true01, p1))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true01: np.ndarray, p1: np.ndarray) -> float:
    """
    PR-AUC (Average Precision) if sklearn is available; else NaN.
    """
    try:
        from sklearn.metrics import average_precision_score
        if len(np.unique(y_true01)) < 2:
            return float("nan")
        return float(average_precision_score(y_true01, p1))
    except Exception:
        return float("nan")


def _action_dist(y_pred_3cls: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_pred_3cls.astype(int), minlength=3).astype(float)
    s = counts.sum()
    return counts / s if s > 0 else np.array([np.nan, np.nan, np.nan], dtype=float)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    q = np.clip(p.astype(float), eps, 1.0)
    q = q / q.sum()
    return float(-np.sum(q * np.log(q)))


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between 2 discrete distributions.
    """
    p = np.clip(p.astype(float), eps, 1.0)
    q = np.clip(q.astype(float), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _util_bucket(u: float) -> str:
    if np.isnan(u):
        return "NA"
    if u < 30.0:
        return "U_0_30"
    if u < 60.0:
        return "U_30_60"
    if u < 90.0:
        return "U_60_90"
    return "U_90_plus"


def _slice_metrics_gate_dir(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Slice metrics for demo + agent triggers:
      - Gate F1/BalAcc by utilization bucket
      - Dir BalAcc by utilization bucket (only on true NONHOLD)
    Looks for s_utilization or utilization.
    """
    util_col = "s_utilization" if "s_utilization" in df.columns else ("utilization" if "utilization" in df.columns else None)
    if util_col is None:
        return {}

    util_vals = pd.to_numeric(df[util_col], errors="coerce").to_numpy()
    buckets = np.array([_util_bucket(float(x)) for x in util_vals], dtype=object)

    out: Dict[str, Any] = {}
    for b in ["U_0_30", "U_30_60", "U_60_90", "U_90_plus", "NA"]:
        m = (buckets == b)
        if not np.any(m):
            continue

        yt = y_true[m]
        yp = y_pred[m]

        # Gate (binary: NONHOLD vs HOLD)
        gt_gate = (yt != 0).astype(int)
        pr_gate = (yp != 0).astype(int)

        tp = int(((gt_gate == 1) & (pr_gate == 1)).sum())
        fp = int(((gt_gate == 0) & (pr_gate == 1)).sum())
        fn = int(((gt_gate == 1) & (pr_gate == 0)).sum())
        tn = int(((gt_gate == 0) & (pr_gate == 0)).sum())

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)

        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal = 0.5 * (tpr + tnr)

        # Direction (binary: CLI vs CLD) on TRUE NONHOLD only
        mask_nonhold = (yt != 0)
        if mask_nonhold.sum() > 0:
            gt_dir = (yt[mask_nonhold] == 1).astype(int)  # 1=CLI, 0=CLD
            pr_dir = (yp[mask_nonhold] == 1).astype(int)

            tp2 = int(((gt_dir == 1) & (pr_dir == 1)).sum())
            fp2 = int(((gt_dir == 0) & (pr_dir == 1)).sum())
            fn2 = int(((gt_dir == 1) & (pr_dir == 0)).sum())
            tn2 = int(((gt_dir == 0) & (pr_dir == 0)).sum())

            tpr2 = tp2 / max(tp2 + fn2, 1)
            tnr2 = tn2 / max(tn2 + fp2, 1)
            dir_bal = 0.5 * (tpr2 + tnr2)
            n_nonhold = int(mask_nonhold.sum())
        else:
            dir_bal = float("nan")
            n_nonhold = 0

        out[b] = {
            "n": int(m.sum()),
            "gate_f1": float(f1),
            "gate_bal_acc": float(bal),
            "dir_bal_acc_on_true_nonhold": float(dir_bal),
            "n_true_nonhold": n_nonhold,
        }

    return out


def _feature_histogram_stats(df: pd.DataFrame, feat_cols: list[str], n_bins: int = 10) -> Dict[str, Any]:
    """
    Store baseline stats per feature for drift detection:
      - quantile bin edges
      - histogram probabilities
      - missing_rate
    """
    stats: Dict[str, Any] = {}
    for c in feat_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        miss = float(np.mean(np.isnan(x)))
        x2 = x[~np.isnan(x)]

        if x2.size < 10:
            stats[c] = {"missing_rate": miss, "bins": None, "hist": None}
            continue

        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(x2, qs)
        edges = np.unique(edges)  # merge duplicates

        if edges.size < 3:
            stats[c] = {"missing_rate": miss, "bins": None, "hist": None}
            continue

        hist, bin_edges = np.histogram(x2, bins=edges)
        hist = hist.astype(float)
        hist = hist / hist.sum() if hist.sum() > 0 else hist

        stats[c] = {"missing_rate": miss, "bins": bin_edges.tolist(), "hist": hist.tolist()}

    return stats


def _psi_from_hist(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-12) -> float:
    e = np.clip(expected.astype(float), eps, 1.0)
    a = np.clip(actual.astype(float), eps, 1.0)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _feature_drift_vs_baseline(df: pd.DataFrame, feat_cols: list[str], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute PSI + missing drift for features where baseline bins/hist exist.
    """
    drift: Dict[str, Any] = {}
    bstats = baseline.get("feature_stats", {})

    for c in feat_cols:
        b = bstats.get(c)
        if not b or not b.get("bins") or not b.get("hist"):
            continue

        bins = np.array(b["bins"], dtype=float)
        expected = np.array(b["hist"], dtype=float)

        x = pd.to_numeric(df[c], errors="coerce").to_numpy()
        miss = float(np.mean(np.isnan(x)))
        x2 = x[~np.isnan(x)]
        if x2.size == 0:
            continue

        hist, _ = np.histogram(x2, bins=bins)
        hist = hist.astype(float)
        hist = hist / hist.sum() if hist.sum() > 0 else hist

        psi = _psi_from_hist(expected, hist)

        drift[c] = {
            "psi": float(psi),
            "missing_rate": miss,
            "missing_rate_baseline": float(b.get("missing_rate", float("nan"))),
            "missing_rate_delta": float(miss - float(b.get("missing_rate", 0.0))),
        }

    return drift


@dataclass
class EvalPipelineConfig:
    test_parquet_3cls: str  # trajectories_test.parquet (3-class action_id)
    out_dir: str = "reports/eval"
    # checkpoints
    gate_ckpt: str = "checkpoints/classification/gate_awac.pt"
    dir_ckpt: str = "checkpoints/classification/dir_awac.pt"
    mag_cli_ckpt: str = "checkpoints/regression/mag_cli_beta.pt"
    mag_cld_ckpt: str = "checkpoints/regression/mag_cld_beta.pt"

    # thresholds (freeze values used in dissertation!)
    gate_thr: float = 0.75
    dir_thr: float = 0.23
    max_pct: float = 50.0

    # optional MLflow logging
    use_mlflow: bool = False
    mlflow_uri: Optional[str] = None
    experiment_name: str = "eval"
    run_name: str = "eval_test"

    # NEW: baseline stats for drift detection (agents read drift.json)
    # If missing, will auto-create baseline from current eval set (demo-friendly).
    baseline_stats_json: Optional[str] = "reports/baseline/baseline_stats.json"


def run_eval(conf: EvalPipelineConfig) -> Dict[str, Any]:
    out = Path(conf.out_dir)
    _ensure_dir(out)

    df = _read_parquet(conf.test_parquet_3cls).copy()
    if "cust_id" not in df.columns:
        raise RuntimeError("test parquet must contain cust_id")
    if "action_id" not in df.columns:
        raise RuntimeError("test parquet must contain action_id (0/1/2 ground truth)")

    # Build features dict per row (s_ columns only)
    feat_cols = [c for c in df.columns if c.startswith("s_")]
    if not feat_cols:
        raise RuntimeError("No s_ feature columns found for evaluation.")

    # Use your production inference (serving) implementation:
    from src.serving.model_loader import init_bundle
    from src.serving.inference import predict_one, InferenceConfig

    gate_path = Path(conf.gate_ckpt).resolve()
    dir_path = Path(conf.dir_ckpt).resolve()
    mag_cli_path = Path(conf.mag_cli_ckpt).resolve()
    mag_cld_path = Path(conf.mag_cld_ckpt).resolve()

    # root ".../checkpoints"
    ckpt_root = str(gate_path.parent.parent)

    # relative paths like "classification/gate_awac.pt"
    gate_file = str(gate_path.relative_to(ckpt_root))
    dir_file = str(dir_path.relative_to(ckpt_root))
    mag_cli_file = str(mag_cli_path.relative_to(ckpt_root))
    mag_cld_file = str(mag_cld_path.relative_to(ckpt_root))


    bundle = init_bundle(
        ckpt_root=ckpt_root,
        device="cpu",
        gate_file=gate_file,
        dir_file=dir_file,
        mag_cli_file=mag_cli_file,
        mag_cld_file=mag_cld_file,
    )

    icfg = InferenceConfig(gate_threshold=conf.gate_thr, dir_threshold=conf.dir_thr, max_pct=conf.max_pct)

    preds = []
    for _, row in df.iterrows():
        feats = {c: float(row[c]) for c in feat_cols}
        prev_limit = float(row.get("s_credit_limit", row.get("credit_limit", 0.0)))
        action, mag_pct, updated, dir_prob, gate_prob = predict_one(
            bundle=bundle,
            features=feats,
            prev_credit_limit=prev_limit,
            next_month="T+1",
            cfg=icfg,
        )
        preds.append((action, mag_pct, updated, gate_prob, dir_prob))

    df["pred_action"] = [p[0] for p in preds]
    df["pred_mag_pct"] = [p[1] for p in preds]
    df["pred_updated_limit"] = [p[2] for p in preds]
    df["pred_gate_prob"] = [p[3] for p in preds]
    df["pred_dir_prob"] = [p[4] for p in preds]

    # Map to 0/1/2
    action_map = {"HOLD": 0, "CLI": 1, "CLD": 2}
    y_true = df["action_id"].astype(int).to_numpy()
    y_pred = df["pred_action"].map(action_map).astype(int).to_numpy()

    cm = _confusion_matrix_3cls(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["GT_HOLD", "GT_CLI", "GT_CLD"], columns=["PRED_HOLD", "PRED_CLI", "PRED_CLD"])
    cm_path = out / "confusion_3cls.csv"
    cm_df.to_csv(cm_path)

    # Gate eval
    gt_gate = (y_true != 0).astype(int)
    pr_gate = (y_pred != 0).astype(int)
    tp = int(((gt_gate == 1) & (pr_gate == 1)).sum())
    fp = int(((gt_gate == 0) & (pr_gate == 1)).sum())
    fn = int(((gt_gate == 1) & (pr_gate == 0)).sum())
    tn = int(((gt_gate == 0) & (pr_gate == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    gate_tpr = tp / max(tp + fn, 1)
    gate_tnr = tn / max(tn + fp, 1)
    gate_bal_acc = 0.5 * (gate_tpr + gate_tnr)

    # Dir eval on true NONHOLD
    mask_nonhold = (y_true != 0)
    gt_dir = (y_true[mask_nonhold] == 1).astype(int)  # 1=CLI, 0=CLD
    pr_dir = (y_pred[mask_nonhold] == 1).astype(int)

    dir_acc = float((gt_dir == pr_dir).mean()) if gt_dir.size else float("nan")
    dir_tp = int(((gt_dir == 1) & (pr_dir == 1)).sum())
    dir_fp = int(((gt_dir == 0) & (pr_dir == 1)).sum())
    dir_fn = int(((gt_dir == 1) & (pr_dir == 0)).sum())
    dir_tn = int(((gt_dir == 0) & (pr_dir == 0)).sum())

    dir_prec = dir_tp / max(dir_tp + dir_fp, 1)
    dir_rec = dir_tp / max(dir_tp + dir_fn, 1)
    dir_f1 = (2 * dir_prec * dir_rec) / max(dir_prec + dir_rec, 1e-12)

    dir_tpr = dir_tp / max(dir_tp + dir_fn, 1)
    dir_tnr = dir_tn / max(dir_tn + dir_fp, 1)
    dir_bal_acc = 0.5 * (dir_tpr + dir_tnr)

    # ---------------------------
    # NEW: Probability-quality metrics (Gate + Direction)
    # ---------------------------
    gate_prob = pd.to_numeric(df["pred_gate_prob"], errors="coerce").fillna(0.0).to_numpy()
    gate_ll = _logloss(gt_gate, gate_prob)
    gate_brier = _brier(gt_gate, gate_prob)
    gate_ece = _ece(gt_gate, gate_prob, n_bins=10)
    gate_auc = _safe_auc(gt_gate, gate_prob)
    gate_pr_auc = _safe_pr_auc(gt_gate, gate_prob)

    dir_prob_all = pd.to_numeric(df["pred_dir_prob"], errors="coerce").fillna(0.0).to_numpy()
    if mask_nonhold.sum() > 0:
        dir_prob = dir_prob_all[mask_nonhold]
        dir_ll = _logloss(gt_dir, dir_prob)
        dir_brier = _brier(gt_dir, dir_prob)
        dir_ece = _ece(gt_dir, dir_prob, n_bins=10)
        dir_auc = _safe_auc(gt_dir, dir_prob)
        dir_pr_auc = _safe_pr_auc(gt_dir, dir_prob)
    else:
        dir_ll = dir_brier = dir_ece = dir_auc = dir_pr_auc = float("nan")

    # ---------------------------
    # NEW: Action behavior metrics (distribution + entropy)
    # ---------------------------
    pred_action_dist = _action_dist(y_pred)
    pred_action_entropy = _entropy(pred_action_dist)

    # ---------------------------
    # NEW: Slice metrics (utilization buckets)
    # ---------------------------
    slices = _slice_metrics_gate_dir(df, y_true=y_true, y_pred=y_pred)

    # Magnitude eval on true CLI/CLD only if magnitude exists in GT
    mag_metrics: Dict[str, Any] = {}
    if "magnitude_pct" in df.columns:
        true_mag = pd.to_numeric(df["magnitude_pct"], errors="coerce").fillna(0.0).to_numpy()
        pred_mag = pd.to_numeric(df["pred_mag_pct"], errors="coerce").fillna(0.0).to_numpy()
        abs_err = np.abs(true_mag - pred_mag)
        sq_err = (true_mag - pred_mag) ** 2
        for name, aid in [("CLI", 1), ("CLD", 2)]:
            m = (y_true == aid)
            ae = abs_err[m]
            rmse = float(np.sqrt(np.nanmean(sq_err[m]))) if ae.size else float("nan")
            mag_metrics[name] = {
                "mae_pp": float(np.nanmean(ae)) if ae.size else float("nan"),
                "rmse_pp": rmse,
                "median_ae": float(np.nanmedian(ae)) if ae.size else float("nan"),
                **_quantiles(ae, (0.90, 0.95)),
                "n": int(ae.size),
            }

    # Keep original keys + add new ones (backward-compatible)
    metrics: Dict[str, Any] = {
        "gate": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_acc": gate_bal_acc,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            # NEW
            "logloss": gate_ll,
            "brier": gate_brier,
            "ece_10bins": gate_ece,
            "roc_auc": gate_auc,
            "pr_auc": gate_pr_auc,
        },
        "dir": {
            "acc_on_true_nonhold": dir_acc,
            "f1": dir_f1,
            "balanced_acc": dir_bal_acc,
            "n_true_nonhold": int(mask_nonhold.sum()),
            # NEW
            "logloss": dir_ll,
            "brier": dir_brier,
            "ece_10bins": dir_ece,
            "roc_auc": dir_auc,
            "pr_auc": dir_pr_auc,
        },
        "confusion_3cls": cm.tolist(),
        "action_behavior": {
            "pred_action_dist": {
                "HOLD": float(pred_action_dist[0]),
                "CLI": float(pred_action_dist[1]),
                "CLD": float(pred_action_dist[2]),
            },
            "pred_action_entropy": float(pred_action_entropy),
        },
        "slices": slices,
        "magnitude": mag_metrics,
    }

    preds_path = out / "preds_test.parquet"
    df.to_parquet(preds_path, index=False)

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # ---------------------------
    # NEW: Drift report for agents (drift.json)
    #   - baseline feature PSI + missing drift
    #   - action JS divergence vs baseline
    # If baseline doesn't exist, create it from current eval set.
    # ---------------------------
    drift_report: Dict[str, Any] = {"baseline_used": False}

    baseline_path = Path(conf.baseline_stats_json) if conf.baseline_stats_json else None
    if baseline_path is not None:
        _ensure_dir(baseline_path.parent)

        if not baseline_path.exists():
            baseline = {
                "created_from": "eval_set_autobaseline",
                "pred_action_dist": {
                    "HOLD": float(pred_action_dist[0]),
                    "CLI": float(pred_action_dist[1]),
                    "CLD": float(pred_action_dist[2]),
                },
                "feature_stats": _feature_histogram_stats(df, feat_cols, n_bins=10),
            }
            baseline_path.write_text(json.dumps(baseline, indent=2))
            drift_report["baseline_used"] = False
            drift_report["baseline_note"] = f"Baseline did not exist. Created baseline at: {str(baseline_path)}"
        else:
            baseline = json.loads(baseline_path.read_text())
            drift_report["baseline_used"] = True

        # Action JS divergence vs baseline
        bdist_dict = baseline.get("pred_action_dist", {})
        bdist = np.array(
            [
                bdist_dict.get("HOLD", np.nan),
                bdist_dict.get("CLI", np.nan),
                bdist_dict.get("CLD", np.nan),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(bdist)) and np.all(np.isfinite(pred_action_dist)):
            drift_report["action_js_divergence"] = _js_divergence(pred_action_dist, bdist)
        else:
            drift_report["action_js_divergence"] = float("nan")

        # Feature PSI drift vs baseline
        drift_report["feature_drift"] = _feature_drift_vs_baseline(df, feat_cols, baseline)

    drift_path = out / "drift.json"
    drift_path.write_text(json.dumps(drift_report, indent=2))

    # Optional MLflow logging
    if conf.use_mlflow:
        import mlflow

        if conf.mlflow_uri:
            mlflow.set_tracking_uri(conf.mlflow_uri)
        mlflow.set_experiment(conf.experiment_name)
        with mlflow.start_run(run_name=conf.run_name):
            mlflow.log_params({"gate_thr": conf.gate_thr, "dir_thr": conf.dir_thr, "max_pct": conf.max_pct})
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(preds_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(drift_path))

            # headline metrics (existing)
            mlflow.log_metric("gate_f1", f1)
            mlflow.log_metric("dir_acc", dir_acc if not np.isnan(dir_acc) else 0.0)

            # NEW headline metrics (agents/drift)
            mlflow.log_metric("gate_brier", gate_brier)
            mlflow.log_metric("gate_logloss", gate_ll)
            mlflow.log_metric("action_entropy", pred_action_entropy)
            js = drift_report.get("action_js_divergence", float("nan"))
            if js == js:  # not NaN
                mlflow.log_metric("action_js", float(js))

    return {
        "preds_parquet": str(preds_path),
        "metrics_json": str(metrics_path),
        "confusion_csv": str(cm_path),
        "drift_json": str(drift_path),
        "metrics": metrics,
        "drift": drift_report,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--out_dir", default="reports/eval")
    args = p.parse_args()

    conf = EvalPipelineConfig(test_parquet_3cls=args.test, out_dir=args.out_dir)
    out = run_eval(conf)
    print("Wrote:", out["metrics_json"])
    print("Wrote:", out["drift_json"])


if __name__ == "__main__":
    main()
