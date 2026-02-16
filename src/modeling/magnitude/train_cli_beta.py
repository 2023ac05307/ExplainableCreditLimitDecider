#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.training.train_mag_beta_core import MagBetaConfig, train_mag_beta


@dataclass
class TrainCLIMagConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str = "checkpoints/mag_cli_beta.pt"

    max_pct: float = 40.0
    epochs: int = 25
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda"
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    early_stop_warmup: int = 2

    # DataLoader speed
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2
    pin_memory: bool = True    


def run_train_cli(conf: TrainCLIMagConfig) -> Dict[str, Any]:
    core = MagBetaConfig(
        train_parquet=conf.train_parquet,
        val_parquet=conf.val_parquet,
        out_ckpt=conf.out_ckpt,
        action="CLI",                 
        max_pct=conf.max_pct,
        epochs=conf.epochs,
        batch_size=conf.batch_size,
        lr=conf.lr,
        weight_decay=conf.weight_decay,
        seed=conf.seed,
        device=conf.device,
        early_stop_patience=conf.early_stop_patience,
        early_stop_min_delta=conf.early_stop_min_delta,
        early_stop_warmup=conf.early_stop_warmup,
        num_workers=conf.num_workers,
        persistent_workers=conf.persistent_workers,
        prefetch_factor=conf.prefetch_factor,
        pin_memory=conf.pin_memory,
    )
    return train_mag_beta(core)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", default="checkpoints/mag_cli_beta.pt")
    p.add_argument("--max-pct", type=float, default=40.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    run_train_cli(
        TrainCLIMagConfig(
            train_parquet=args.train,
            val_parquet=args.val,
            out_ckpt=args.out,
            max_pct=args.max_pct,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
