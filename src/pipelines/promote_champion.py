#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import json

from src.monitoring.drift_metrics import (
    compute_drift_report,
    save_drift_report,
    DriftThresholds,
)


@dataclass
class PromoteConfig:
    ckpt_dir: str = "checkpoints"
    eval_metrics_json: str = "reports/eval/metrics.json"
    champion_file: str = "checkpoints/CHAMPION.json"

    # promotion rules
    min_gate_f1: float = 0.70
    min_dir_acc: float = 0.80

    # ---- Drift gate (production governance)
    enable_drift_gate: bool = True
    drift_reference_path: str = "rl_dataset/splits/trajectories_val.parquet"
    drift_current_path: str = "rl_dataset/splits/trajectories_aug"  # can be parquet dir
    drift_feature_prefix: str = "s_"
    drift_report_json: str = "reports/drift/drift_summary.json"
    drift_thresholds: DriftThresholds = DriftThresholds(psi_low=0.10, psi_medium=0.20, psi_high=0.30)
    block_on_drift_severity: str = "high"  # none | low | medium | high
    drift_sample_ref: int = 200_000
    drift_sample_cur: int = 200_000

    # MLflow optional
    use_mlflow_registry: bool = False
    mlflow_uri: Optional[str] = None
    registered_model_name: str = "creditlimit_policy"


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _severity_rank(sev: str) -> int:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return order.get(sev, 0)


def _run_drift_gate(conf: PromoteConfig) -> Dict[str, Any]:
    """
    Compute drift report and decide whether to block promotion.
    Returns dict with drift report path + severity.
    """
    rep = compute_drift_report(
        reference_path=conf.drift_reference_path,
        current_path=conf.drift_current_path,
        feature_prefix=conf.drift_feature_prefix,
        thresholds=conf.drift_thresholds,
        sample_ref=conf.drift_sample_ref,
        sample_cur=conf.drift_sample_cur,
    )
    save_drift_report(rep, conf.drift_report_json)

    block_threshold = conf.block_on_drift_severity
    should_block = _severity_rank(rep.overall_severity) >= _severity_rank(block_threshold)

    return {
        "enabled": True,
        "reference": conf.drift_reference_path,
        "current": conf.drift_current_path,
        "report": conf.drift_report_json,
        "overall_severity": rep.overall_severity,
        "summary": rep.summary,
        "blocked": bool(should_block),
        "block_threshold": block_threshold,
    }


def _local_promote(conf: PromoteConfig, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes a CHAMPION.json pointer to current checkpoint set.
    Applies drift gate (optional) + metric gates.
    """
    drift_info: Dict[str, Any] = {"enabled": False}

    if conf.enable_drift_gate:
        drift_info = _run_drift_gate(conf)
        if drift_info.get("blocked"):
            return {
                "promoted": False,
                "reason": "blocked_by_drift",
                "drift": drift_info,
                "ckpt_dir": conf.ckpt_dir,
            }

    gate_f1 = float(metrics["gate"]["f1"])
    dir_acc = float(metrics["dir"]["acc_on_true_nonhold"])

    ok = (gate_f1 >= conf.min_gate_f1) and (dir_acc >= conf.min_dir_acc)

    out = {
        "promoted": bool(ok),
        "reason": "ok" if ok else "metrics_below_threshold",
        "gate_f1": gate_f1,
        "dir_acc": dir_acc,
        "ckpt_dir": conf.ckpt_dir,
        "rules": {"min_gate_f1": conf.min_gate_f1, "min_dir_acc": conf.min_dir_acc},
        "drift": drift_info,
    }

    if ok:
        Path(conf.champion_file).parent.mkdir(parents=True, exist_ok=True)
        Path(conf.champion_file).write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out


def _mlflow_promote(conf: PromoteConfig, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal MLflow promotion example.
    Still enforces drift gate (optional).
    """
    drift_info: Dict[str, Any] = {"enabled": False}
    if conf.enable_drift_gate:
        drift_info = _run_drift_gate(conf)
        if drift_info.get("blocked"):
            return {"promoted": False, "reason": "blocked_by_drift", "drift": drift_info}

    import mlflow

    if conf.mlflow_uri:
        mlflow.set_tracking_uri(conf.mlflow_uri)

    gate_f1 = float(metrics["gate"]["f1"])
    dir_acc = float(metrics["dir"]["acc_on_true_nonhold"])
    ok = (gate_f1 >= conf.min_gate_f1) and (dir_acc >= conf.min_dir_acc)

    if not ok:
        return {
            "promoted": False,
            "reason": "metrics_below_threshold",
            "gate_f1": gate_f1,
            "dir_acc": dir_acc,
            "drift": drift_info,
        }

    with mlflow.start_run(run_name="promote_champion"):
        mlflow.log_params(
            {
                "min_gate_f1": conf.min_gate_f1,
                "min_dir_acc": conf.min_dir_acc,
                "drift_gate": conf.enable_drift_gate,
                "drift_severity": drift_info.get("overall_severity"),
            }
        )
        mlflow.log_metric("gate_f1", gate_f1)
        mlflow.log_metric("dir_acc", dir_acc)

        # Optionally log drift report
        if conf.enable_drift_gate and Path(conf.drift_report_json).exists():
            mlflow.log_artifact(conf.drift_report_json, artifact_path="drift")

        # Log model checkpoints directory
        mlflow.log_artifacts(conf.ckpt_dir, artifact_path="champion_checkpoints")

    return {"promoted": True, "gate_f1": gate_f1, "dir_acc": dir_acc, "mode": "mlflow_artifacts", "drift": drift_info}


def run_promote(conf: PromoteConfig) -> Dict[str, Any]:
    metrics = _read_json(conf.eval_metrics_json)
    if conf.use_mlflow_registry:
        return _mlflow_promote(conf, metrics)
    return _local_promote(conf, metrics)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="reports/eval/metrics.json")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--champion_file", default="checkpoints/CHAMPION.json")

    # drift args
    p.add_argument("--enable_drift_gate", action="store_true")
    p.add_argument("--drift_ref", default="rl_dataset/splits/trajectories_val.parquet")
    p.add_argument("--drift_cur", default="rl_dataset/splits/trajectories_aug")
    p.add_argument("--drift_report", default="reports/drift/drift_summary.json")
    p.add_argument("--drift_block_on", default="high", choices=["none", "low", "medium", "high"])

    args = p.parse_args()

    conf = PromoteConfig(
        ckpt_dir=args.ckpt_dir,
        eval_metrics_json=args.metrics,
        champion_file=args.champion_file,
        enable_drift_gate=bool(args.enable_drift_gate),
        drift_reference_path=args.drift_ref,
        drift_current_path=args.drift_cur,
        drift_report_json=args.drift_report,
        block_on_drift_severity=args.drift_block_on,
    )

    out = run_promote(conf)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
