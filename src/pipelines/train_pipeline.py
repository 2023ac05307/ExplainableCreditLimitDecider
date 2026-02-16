"""
train_pipeline.py
-----------------
Trains:
  1) Gate model   (HOLD vs NONHOLD)       using gated_train/gated_val
  2) Dir model    (CLD vs CLI on NONHOLD) using dir_train/dir_val
  3) Magnitude models (Beta regression):
        - CLI magnitude model (fixed action CLI)
        - CLD magnitude model (fixed action CLD)

IMPORTANT:
Magnitude training must be from 3-class trajectories where:
  action_id: 0=HOLD, 1=CLI, 2=CLD
So we use trajectories_train_aug_3cls + trajectories_val_3cls.

Run:
  python -m src.pipelines.train_pipeline --config configs/paths.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from src.monitoring.push_metrics import push_job_status
import time

import yaml

# ✅ Repo-aligned imports (your trainers are under src/modeling/*)
from src.modeling.gate.train_gate_awac import GateTrainConfig, run as run_gate
from src.modeling.direction.train_dir_awac import DirTrainConfig, run as run_dir
from src.modeling.magnitude.train_cli_beta import TrainCLIMagConfig, run_train_cli
from src.modeling.magnitude.train_cld_beta import TrainCLDMagConfig, run_train_cld


# ---------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------
@dataclass
class TrainingPipelineConfig:
    # ---- classification datasets ----
    gated_train: str
    gated_val: str
    dir_train: str
    dir_val: str

    # ---- magnitude trajectories (3-class) ----
    traj_train_aug_3cls: str
    traj_val_3cls: str

    # ---- outputs ----
    ckpt_dir: str = "checkpoints"

    # ---- toggles ----
    do_gate: bool = False
    do_dir: bool = True
    do_mag_cli: bool = True
    do_mag_cld: bool = True

    # ---- shared ----
    seed: int = 42
    device: str = "cuda"


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(cfg: Dict[str, Any]) -> TrainingPipelineConfig:
    ckpt_root = (cfg.get("checkpoints", {}).get("root")) or "checkpoints"

    gate_dir_dir = cfg.get("gate_dir_dir", "rl_dataset/gate_dir")
    splits_dir = cfg.get("splits_dir", "rl_dataset/splits")

    gated_train = str(Path(gate_dir_dir) / "gated_train.parquet")
    gated_val   = str(Path(gate_dir_dir) / "gated_val.parquet")
    dir_train   = str(Path(gate_dir_dir) / "dir_train.parquet")
    dir_val     = str(Path(gate_dir_dir) / "dir_val.parquet")

    # data_pipeline writes augmented train as a parquet dataset path "trajectories_aug" (may be a folder)
    traj_train_aug_3cls = str(Path(splits_dir) / "trajectories_aug")
    traj_val_3cls = str(Path(splits_dir) / "trajectories_val.parquet")

    seed = int(cfg.get("seed", 42))
    device = str(cfg.get("device", "cuda"))

    return TrainingPipelineConfig(
        gated_train=gated_train,
        gated_val=gated_val,
        dir_train=dir_train,
        dir_val=dir_val,
        traj_train_aug_3cls=traj_train_aug_3cls,
        traj_val_3cls=traj_val_3cls,
        ckpt_dir=ckpt_root,
        seed=seed,
        device=device,
    )



def run_training(conf: TrainingPipelineConfig) -> Dict[str, Any]:
    ckpt_root = Path(conf.ckpt_dir)
    (ckpt_root / "classification").mkdir(parents=True, exist_ok=True)
    (ckpt_root / "regression").mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {}

    # -----------------------------
    # 1) Gate
    # -----------------------------
    if conf.do_gate:
        gate_ckpt = str(ckpt_root / "classification" / "gate_awac.pt")
        run_gate(
            GateTrainConfig(
                train_parquet=conf.gated_train,
                val_parquet=conf.gated_val,
                out_ckpt=gate_ckpt,
                seed=conf.seed,
            )
        )
        outputs["gate_ckpt"] = gate_ckpt

    # -----------------------------
    # 2) Dir
    # -----------------------------
    if conf.do_dir:
        dir_ckpt = str(ckpt_root / "classification" / "dir_awac.pt")
        run_dir(
            DirTrainConfig(
                train_parquet=conf.dir_train,
                val_parquet=conf.dir_val,
                out_ckpt=dir_ckpt,
                seed=conf.seed,
            )
        )
        outputs["dir_ckpt"] = dir_ckpt

    # -----------------------------
    # 3) Magnitude - CLI (fixed action)
    # -----------------------------
    if conf.do_mag_cli:
        mag_cli_ckpt = str(ckpt_root / "regression" / "mag_cli_beta.pt")
        cli_metrics = run_train_cli(
            TrainCLIMagConfig(
                train_parquet=conf.traj_train_aug_3cls,
                val_parquet=conf.traj_val_3cls,
                out_ckpt=mag_cli_ckpt,
                seed=conf.seed,
                device=conf.device,
            )
        )
        outputs["mag_cli_ckpt"] = mag_cli_ckpt
        outputs["mag_cli_metrics"] = cli_metrics

    # -----------------------------
    # 4) Magnitude - CLD (fixed action)
    # -----------------------------
    if conf.do_mag_cld:
        mag_cld_ckpt = str(ckpt_root / "regression" / "mag_cld_beta.pt")
        cld_metrics = run_train_cld(
            TrainCLDMagConfig(
                train_parquet=conf.traj_train_aug_3cls,
                val_parquet=conf.traj_val_3cls,
                out_ckpt=mag_cld_ckpt,
                seed=conf.seed,
                device=conf.device,
            )
        )
        outputs["mag_cld_ckpt"] = mag_cld_ckpt
        outputs["mag_cld_metrics"] = cld_metrics

    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/paths.yaml)")
    p.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    p.add_argument("--seed", type=int, default=None, help="Override seed")

    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-dir", action="store_true")
    p.add_argument("--no-mag-cli", action="store_true")
    p.add_argument("--no-mag-cld", action="store_true")
    return p.parse_args()


def main() -> None:
    import time

    args = parse_args()
    cfg = _read_yaml(args.config)
    conf = build_config(cfg)

    if args.device is not None:
        conf.device = args.device
    if args.seed is not None:
        conf.seed = args.seed

    conf.do_gate = not args.no_gate
    conf.do_dir = not args.no_dir
    conf.do_mag_cli = not args.no_mag_cli
    conf.do_mag_cld = not args.no_mag_cld

    t0 = time.time()
    ok = False

    try:
        out = run_training(conf)
        ok = True

        print("✅ Training pipeline complete:")
        for k, v in out.items():
            print(f"  - {k}: {v}")

    finally:
        push_job_status(
            job_name="train_pipeline",
            ok=ok,
            duration_s=time.time() - t0,
            extra={
                "trained_gate": int(conf.do_gate),
                "trained_dir": int(conf.do_dir),
                "trained_mag_cli": int(conf.do_mag_cli),
                "trained_mag_cld": int(conf.do_mag_cld),
            },
        )


if __name__ == "__main__":
    main()
