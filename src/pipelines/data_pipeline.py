#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/pipelines/data_pipeline.py
------------------------------

"""

from __future__ import annotations
from src.monitoring.push_metrics import push_job_status
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from src.monitoring.sanity_checks import (
    run_sanity_checks,
    save_sanity_report,
    SanityThresholds,
)


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def _now() -> float:
    return time.time()


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def exists_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_step(name: str, fn, *args, **kwargs):
    log(f"\n=== [{name}] START ===")
    t0 = _now()
    out = fn(*args, **kwargs)
    dt = _now() - t0
    log(f"=== [{name}] DONE in {dt:.2f}s ===")
    return out


# ---------------------------------------------------------------------
# Sanity gate helpers
# ---------------------------------------------------------------------
def _sanity_gate(
    dataset_path: str,
    report_path: str,
    *,
    feature_prefix: str = "s_",
    thresholds: Optional[SanityThresholds] = None,
) -> None:
    rep = run_sanity_checks(dataset_path, thresholds=thresholds, feature_prefix=feature_prefix)
    save_sanity_report(rep, report_path)

    if not rep.passed:
        raise RuntimeError(
            f"Sanity checks FAILED for {dataset_path}. "
            f"See report: {report_path} | failures={rep.failures}"
        )
    if rep.warnings:
        log(f"[WARN] Sanity warnings for {dataset_path}: {rep.warnings}")


# Different dataset types may have different columns.
# Snapshots may NOT have s_* feature columns. Trajectories/Gate/Dir should.
SNAPSHOT_THRESHOLDS = SanityThresholds(
    min_rows=10_000,
    max_overall_missing_rate=0.40,          # snapshots can be sparse early
    max_missing_rate_per_feature=0.90,      # not strict on per-feature here
    enforce_unique_key=False,               # snapshots may have multiple rows per cust/month depending on design
)

TRAJ_THRESHOLDS = SanityThresholds(
    min_rows=50_000,
    max_overall_missing_rate=0.25,
    max_missing_rate_per_feature=0.60,
    enforce_unique_key=True,                # trajectories should be unique per cust_id/month (or your key)
)

GATE_DIR_THRESHOLDS = SanityThresholds(
    min_rows=20_000,
    max_overall_missing_rate=0.20,
    max_missing_rate_per_feature=0.50,
    enforce_unique_key=False,               # gate/dir are row-based samples; uniqueness may not apply
)


# ---------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------
@dataclass
class DataPipelineConfig:
    sim_dir: Path
    dataset_dir: Path
    splits_dir: Path
    gate_dir_dir: Path

    base_snapshots_path: Path
    final_snapshots_path: Path
    traj_all_path: Path
    traj_strict_path: Path

    train_traj_path: Path
    val_traj_path: Path
    test_traj_path: Path
    train_aug_path: Path

    gated_train_path: Path
    gated_val_path: Path
    dir_train_path: Path
    dir_val_path: Path

    seed: int = 42
    val_frac: float = 0.10
    test_frac: float = 0.10
    batch_rows: int = 200_000
    compression: str = "snappy"

    do_split: bool = True
    do_augment: bool = True
    do_gate_dir: bool = True

    force: bool = False


def build_config(cfg: Dict[str, Any], force: bool) -> DataPipelineConfig:
    sim_dir = Path(cfg["sim_dir"])
    dataset_dir = Path(cfg.get("dataset_dir", "rl_dataset"))

    splits_dir = Path(cfg.get("splits_dir", str(dataset_dir / "splits")))
    gate_dir_dir = Path(cfg.get("gate_dir_dir", str(dataset_dir / "gate_dir")))

    base_snapshots_path = dataset_dir / "snapshots_base_all_years.parquet"
    final_snapshots_path = dataset_dir / "snapshots_all_years.parquet"
    traj_all_path = dataset_dir / "trajectories_all.parquet"
    traj_strict_path = dataset_dir / "trajectories_strict.parquet"

    train_traj_path = splits_dir / "trajectories_train.parquet"
    val_traj_path = splits_dir / "trajectories_val.parquet"
    test_traj_path = splits_dir / "trajectories_test.parquet"
    train_aug_path = splits_dir / "trajectories_aug"

    gated_train_path = gate_dir_dir / "gated_train.parquet"
    gated_val_path = gate_dir_dir / "gated_val.parquet"
    dir_train_path = gate_dir_dir / "dir_train.parquet"
    dir_val_path = gate_dir_dir / "dir_val.parquet"

    return DataPipelineConfig(
        sim_dir=sim_dir,
        dataset_dir=dataset_dir,
        splits_dir=splits_dir,
        gate_dir_dir=gate_dir_dir,

        base_snapshots_path=base_snapshots_path,
        final_snapshots_path=final_snapshots_path,
        traj_all_path=traj_all_path,
        traj_strict_path=traj_strict_path,

        train_traj_path=train_traj_path,
        val_traj_path=val_traj_path,
        test_traj_path=test_traj_path,
        train_aug_path=train_aug_path,

        gated_train_path=gated_train_path,
        gated_val_path=gated_val_path,
        dir_train_path=dir_train_path,
        dir_val_path=dir_val_path,

        seed=int(cfg.get("seed", 42)),
        val_frac=float(cfg.get("val_frac", 0.10)),
        test_frac=float(cfg.get("test_frac", 0.10)),
        batch_rows=int(cfg.get("batch_rows", 200_000)),
        compression=str(cfg.get("compression", "snappy")),

        do_split=bool(cfg.get("do_split", True)),
        do_augment=bool(cfg.get("do_augment", True)),
        do_gate_dir=bool(cfg.get("do_gate_dir", True)),

        force=force,
    )


# ---------------------------------------------------------------------
# Step: Build snapshots
# ---------------------------------------------------------------------
def step_build_monthly_snapshots(conf: DataPipelineConfig) -> None:
    ensure_dir(conf.dataset_dir)

    if exists_nonempty(conf.base_snapshots_path) and not conf.force:
        log(f"[SKIP] base snapshots exist: {conf.base_snapshots_path}")
        return

    from src.data_engineering.build_monthly_snapshots import build_base_monthly_snapshots
    build_base_monthly_snapshots(str(conf.sim_dir), str(conf.dataset_dir))

    if not exists_nonempty(conf.base_snapshots_path):
        raise RuntimeError(f"Expected base snapshots not found: {conf.base_snapshots_path}")

    # Sanity gate (snapshots may not have s_* columns, so prefix is "" and relaxed thresholds)
    _sanity_gate(
        str(conf.base_snapshots_path),
        "reports/sanity/snapshots_base.json",
        feature_prefix="",  # do not enforce s_* on snapshots
        thresholds=SNAPSHOT_THRESHOLDS,
    )


# ---------------------------------------------------------------------
# Step: Feature engineering
# ---------------------------------------------------------------------
def step_feature_engineering(conf: DataPipelineConfig) -> None:
    ensure_dir(conf.dataset_dir)

    if exists_nonempty(conf.final_snapshots_path) and not conf.force:
        log(f"[SKIP] final snapshots exist: {conf.final_snapshots_path}")
        return

    if not exists_nonempty(conf.base_snapshots_path):
        raise FileNotFoundError(f"Missing base snapshots: {conf.base_snapshots_path}")

    from src.data_engineering.feature_engineering import feature_engineer_snapshots
    feature_engineer_snapshots(str(conf.base_snapshots_path), str(conf.dataset_dir))

    if not exists_nonempty(conf.final_snapshots_path):
        raise RuntimeError(f"Expected final snapshots not found: {conf.final_snapshots_path}")

    _sanity_gate(
        str(conf.final_snapshots_path),
        "reports/sanity/snapshots_final.json",
        feature_prefix="",
        thresholds=SNAPSHOT_THRESHOLDS,
    )


# ---------------------------------------------------------------------
# Step: Build trajectories
# ---------------------------------------------------------------------
def step_build_trajectories(conf: DataPipelineConfig) -> None:
    ensure_dir(conf.dataset_dir)

    if exists_nonempty(conf.traj_all_path) and exists_nonempty(conf.traj_strict_path) and not conf.force:
        log(f"[SKIP] trajectories exist: {conf.traj_all_path} / {conf.traj_strict_path}")
        return

    if not exists_nonempty(conf.final_snapshots_path):
        raise FileNotFoundError(f"Missing final snapshots: {conf.final_snapshots_path}")

    from src.data_engineering.build_trajectories import build_trajectories_from_snapshot_file
    build_trajectories_from_snapshot_file(str(conf.final_snapshots_path), str(conf.dataset_dir))

    if not exists_nonempty(conf.traj_all_path) or not exists_nonempty(conf.traj_strict_path):
        raise RuntimeError(f"Expected trajectories not found: {conf.traj_all_path} / {conf.traj_strict_path}")

    # Trajectory sanity (strict is what you use for splits)
    _sanity_gate(
        str(conf.traj_strict_path),
        "reports/sanity/trajectories_strict.json",
        feature_prefix="s_",
        thresholds=TRAJ_THRESHOLDS,
    )


# ---------------------------------------------------------------------
# Step: Split
# ---------------------------------------------------------------------
def step_split(conf: DataPipelineConfig) -> None:
    if not conf.do_split:
        log("[SKIP] do_split=false")
        return

    ensure_dir(conf.splits_dir)

    if (
        exists_nonempty(conf.train_traj_path)
        and exists_nonempty(conf.val_traj_path)
        and exists_nonempty(conf.test_traj_path)
        and not conf.force
    ):
        log(f"[SKIP] split outputs exist under: {conf.splits_dir}")
        return

    if not exists_nonempty(conf.traj_strict_path):
        raise FileNotFoundError(f"Missing strict trajectories: {conf.traj_strict_path}")

    from src.data_engineering.splitting import run_split

    train_frac = 1.0 - conf.val_frac - conf.test_frac
    if train_frac <= 0:
        raise ValueError(f"Invalid split fractions: val_frac={conf.val_frac}, test_frac={conf.test_frac}")

    run_split(
        in_parquet=str(conf.traj_strict_path),
        out_dir=str(conf.splits_dir),
        group_col="cust_id",
        label_col="action_id",
        seed=int(conf.seed),
        train_frac=float(train_frac),
        val_frac=float(conf.val_frac),
        test_frac=float(conf.test_frac),
        compression=str(conf.compression),
    )

    if not (
        exists_nonempty(conf.train_traj_path)
        and exists_nonempty(conf.val_traj_path)
        and exists_nonempty(conf.test_traj_path)
    ):
        raise RuntimeError("Split step completed but expected outputs not found.")

    _sanity_gate(str(conf.train_traj_path), "reports/sanity/trajectories_train.json", thresholds=TRAJ_THRESHOLDS)
    _sanity_gate(str(conf.val_traj_path), "reports/sanity/trajectories_val.json", thresholds=TRAJ_THRESHOLDS)
    _sanity_gate(str(conf.test_traj_path), "reports/sanity/trajectories_test.json", thresholds=TRAJ_THRESHOLDS)


# ---------------------------------------------------------------------
# Step: Augment train
# ---------------------------------------------------------------------
def step_augment_train(conf: DataPipelineConfig) -> None:
    if not conf.do_augment:
        log("[SKIP] do_augment=false")
        return

    ensure_dir(conf.splits_dir)

    aug_dir = Path(conf.train_aug_path)
    summary_json = aug_dir / "_summary.json"
    any_part = next(aug_dir.glob("part-*.parquet"), None) if aug_dir.exists() else None

    if (summary_json.exists() or any_part is not None) and not conf.force:
        log(f"[SKIP] train augmentation dataset exists: {aug_dir}")
        return

    if not exists_nonempty(conf.train_traj_path):
        raise FileNotFoundError(f"Missing train split: {conf.train_traj_path}")

    from src.data_engineering.augment_trajectories import run_augment_counterfactual

    if conf.force and aug_dir.exists():
        for p in aug_dir.glob("*"):
            if p.is_file():
                p.unlink()
    ensure_dir(aug_dir)

    run_augment_counterfactual(
        in_traj=str(conf.train_traj_path),
        out_dir=str(aug_dir),
        cli_mult=10.0,
        cld_mult=3.0,
        cli_variants=1,
        cld_variants=1,
        seed=123,
        compression=str(conf.compression),
    )

    any_part = next(aug_dir.glob("part-*.parquet"), None)
    if any_part is None:
        raise RuntimeError(f"Augment step completed but no part-*.parquet files found in: {aug_dir}")

    # Sanity gate on augmented dataset directory
    _sanity_gate(
        str(aug_dir),
        "reports/sanity/trajectories_aug.json",
        thresholds=TRAJ_THRESHOLDS,
    )


# ---------------------------------------------------------------------
# Step: Prepare Gate + Dir datasets
# ---------------------------------------------------------------------
def step_prepare_gate_dir(conf: DataPipelineConfig) -> None:
    if not conf.do_gate_dir:
        log("[SKIP] do_gate_dir=false")
        return

    ensure_dir(conf.gate_dir_dir)

    if (
        exists_nonempty(conf.gated_train_path)
        and exists_nonempty(conf.gated_val_path)
        and exists_nonempty(conf.dir_train_path)
        and exists_nonempty(conf.dir_val_path)
        and not conf.force
    ):
        log(f"[SKIP] gate/dir outputs exist under: {conf.gate_dir_dir}")
        return

    train_aug = Path(conf.train_aug_path)
    val_path = Path(conf.val_traj_path)

    if not train_aug.exists():
        raise FileNotFoundError(f"Missing augmented train input: {train_aug}")
    if train_aug.is_dir():
        part = next(train_aug.glob("part-*.parquet"), None)
        if part is None:
            raise FileNotFoundError(f"Augmented train dir has no part-*.parquet files: {train_aug}")
    else:
        if not exists_nonempty(train_aug):
            raise FileNotFoundError(f"Augmented train parquet is empty/missing: {train_aug}")

    if not exists_nonempty(val_path):
        raise FileNotFoundError(f"Missing val trajectories: {val_path}")

    from src.data_engineering.prepare_gate_dir_datasets import run_prepare_gate_dir

    run_prepare_gate_dir(
        merged_train=str(train_aug),
        merged_test=str(val_path),
        out_dir=str(conf.gate_dir_dir),
        batch_rows=int(conf.batch_rows),
        compression=str(conf.compression),
    )

    missing = [
        p
        for p in [conf.gated_train_path, conf.gated_val_path, conf.dir_train_path, conf.dir_val_path]
        if not exists_nonempty(p)
    ]
    if missing:
        raise RuntimeError(f"Gate/Dir step completed but missing outputs: {missing}")

    _sanity_gate(str(conf.gated_train_path), "reports/sanity/gated_train.json", thresholds=GATE_DIR_THRESHOLDS)
    _sanity_gate(str(conf.gated_val_path), "reports/sanity/gated_val.json", thresholds=GATE_DIR_THRESHOLDS)
    _sanity_gate(str(conf.dir_train_path), "reports/sanity/dir_train.json", thresholds=GATE_DIR_THRESHOLDS)
    _sanity_gate(str(conf.dir_val_path), "reports/sanity/dir_val.json", thresholds=GATE_DIR_THRESHOLDS)


# ---------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------
def run_pipeline(conf: DataPipelineConfig) -> None:
    run_step("01_BUILD_MONTHLY_SNAPSHOTS", step_build_monthly_snapshots, conf)
    run_step("02_FEATURE_ENGINEERING", step_feature_engineering, conf)
    run_step("03_BUILD_TRAJECTORIES", step_build_trajectories, conf)
    run_step("04_SPLIT_CUSTOMERS", step_split, conf)
    run_step("05_AUGMENT_TRAIN", step_augment_train, conf)
    run_step("06_PREPARE_GATE_DIR", step_prepare_gate_dir, conf)


def main():
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/paths.yaml")
    ap.add_argument("--force", action="store_true", help="Rebuild outputs even if they exist")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    conf = build_config(cfg, force=args.force)

    t0 = time.time()
    ok = False

    try:
        log("=== DATA PIPELINE CONFIG ===")
        log(f"sim_dir        : {conf.sim_dir}")
        log(f"dataset_dir    : {conf.dataset_dir}")
        log(f"splits_dir     : {conf.splits_dir}")
        log(f"gate_dir_dir   : {conf.gate_dir_dir}")

        run_pipeline(conf)
        log("\n✅ Data pipeline complete.")
        ok = True

    finally:
        push_job_status(
            job_name="data_pipeline",
            ok=ok,
            duration_s=time.time() - t0,
        )


if __name__ == "__main__":
    main()
