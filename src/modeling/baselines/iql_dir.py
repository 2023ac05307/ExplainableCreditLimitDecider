#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iql_dir.py
----------
Baseline IQL for Direction (CLD vs CLI) on TRUE NONHOLD only.
- Uses repo DirActorCritic (dict outputs: logit, q, v)
- Config-driven train_loop(conf)
- Option-A checkpoint: ckpt["model"] stores nn.Module
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.modeling.direction.model import DirActorCritic, DirModelConfig
from src.modeling.data.datasets import TrajDatasetDIR
from src.modeling.utils.seed import set_seed
from src.modeling.utils.sampler import build_weighted_sampler
from src.modeling.utils.ema import ema_update
from src.modeling.utils.eval_dir import eval_val_dir_style

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * diff.pow(2)).mean()


@dataclass
class DirIQLConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str = "checkpoints/baselines/dir_iql.pt"

    warmup_epochs: int = 5
    iql_epochs: int = 20

    batch_size: int = 4096
    lr_warmup: float = 2e-4
    lr_iql: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    seed: int = 42

    # IQL knobs
    expectile_tau: float = 0.7
    awr_temp: float = 2.0
    adv_clip: float = 2.0
    w_adv_clip: float = 0.7
    huber_delta: float = 1.0
    ema: float = 0.995

    q_coef: float = 1.0
    v_coef: float = 1.0
    pi_coef: float = 1.0
    ent_coef: float = 0.0

    reward_clip: float = 10.0
    reward_scale: float = 100.0

    # sampling (0=CLD, 1=CLI)
    mix_cld: float = 0.50
    mix_cli: float = 0.50
    boost_cld: float = 1.0
    boost_cli: float = 1.0

    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_steps: int = 19

    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    early_stop_warmup: int = 2


def train_loop(conf: DirIQLConfig) -> Dict[str, Any]:
    set_seed(conf.seed)

    ds_tr = TrajDatasetDIR(conf.train_parquet, scaler=None)
    scaler = {"mu": ds_tr.mu, "sd": ds_tr.sd}
    ds_va = TrajDatasetDIR(conf.val_parquet, scaler=scaler)

    obs_dim = int(ds_tr.s.shape[1])
    mu_t = torch.as_tensor(ds_tr.mu, device=DEVICE, dtype=torch.float32)
    sd_t = torch.as_tensor(ds_tr.sd, device=DEVICE, dtype=torch.float32)
    thresholds = np.linspace(conf.thr_min, conf.thr_max, conf.thr_steps).astype(np.float32)

    print(f"TRAIN: {len(ds_tr):,} | obs_dim={obs_dim} | device={DEVICE}")
    print("TRAIN counts:", {0: int(ds_tr.action_counts[0]), 1: int(ds_tr.action_counts[1])})
    print(f"VAL:   {len(ds_va):,}")
    print("VAL counts:", {0: int(ds_va.action_counts[0]), 1: int(ds_va.action_counts[1])})
    print("Label map: 0=CLD, 1=CLI")

    sampler = build_weighted_sampler(
        ds_tr,
        mix0=conf.mix_cld, mix1=conf.mix_cli,
        boost0=conf.boost_cld, boost1=conf.boost_cli
    )

    tr_loader = DataLoader(
        ds_tr, batch_size=conf.batch_size, sampler=sampler,
        drop_last=True, pin_memory=(DEVICE == "cuda"), num_workers=0
    )
    va_loader = DataLoader(
        ds_va, batch_size=conf.batch_size, shuffle=False,
        drop_last=False, pin_memory=(DEVICE == "cuda"), num_workers=0
    )

    model_cfg = DirModelConfig(obs_dim=obs_dim)
    model = DirActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target = DirActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    os.makedirs(os.path.dirname(conf.out_ckpt) or ".", exist_ok=True)

    best_f1, best_thr, best_epoch = -1.0, 0.5, 0
    epochs_since_improve = 0

    def save_bundle(epoch_global: int, val_best: Dict[str, Any]) -> None:
        torch.save(
            {
                "task": "baseline_dir_iql",
                "model": model,  # ✅ Option A
                "state_dict": model.state_dict(),
                "target_state_dict": target.state_dict(),

                "obs_dim": obs_dim,
                "state_cols": ds_tr.state_cols,
                "scaler_mean": ds_tr.mu,
                "scaler_std": ds_tr.sd,
                "train_action_counts": ds_tr.action_counts,
                "val_action_counts": ds_va.action_counts,

                "best_val_f1": float(val_best["f1"]),
                "best_thr": float(val_best["thr"]),
                "best_epoch": int(epoch_global),
                "label_map": {0: "CLD", 1: "CLI"},

                "model_config": model_cfg.__dict__,
                "train_config": conf.__dict__,
                "ckpt_format": "option_a",
            },
            conf.out_ckpt,
        )

    opt_warm = torch.optim.AdamW(model.parameters(), lr=conf.lr_warmup, weight_decay=conf.weight_decay)
    opt_iql = torch.optim.AdamW(model.parameters(), lr=conf.lr_iql, weight_decay=conf.weight_decay)

    def run_phase(name: str, epochs: int, mode: str, opt: torch.optim.Optimizer, offset: int) -> None:
        nonlocal best_f1, best_thr, best_epoch, epochs_since_improve

        for ep in range(1, epochs + 1):
            model.train()
            train_pi = []

            for s_np, a_np, r_np, s1_np, done_np, w_np in tr_loader:
                s_raw = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32)
                s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
                a = torch.as_tensor(a_np, device=DEVICE, dtype=torch.long)  # 0/1 (CLD/CLI)
                y = a.float()
                r = torch.as_tensor(r_np, device=DEVICE, dtype=torch.float32)
                done = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
                sw = torch.as_tensor(w_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

                s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=0.0, neginf=0.0)
                s1_raw = torch.nan_to_num(s1_raw, nan=0.0, posinf=0.0, neginf=0.0)
                r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

                s = (s_raw - mu_t) / sd_t
                s1 = (s1_raw - mu_t) / sd_t

                out = model(s)
                logit = out["logit"]
                q = out["q"]
                v = out["v"]

                bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                den = sw.sum().clamp_min(1.0)

                if mode == "bce":
                    pi_loss = (sw * bce).sum() / den
                    loss = pi_loss
                else:
                    r2 = r / max(1e-6, conf.reward_scale)
                    r2 = torch.clamp(r2, -conf.reward_clip, conf.reward_clip)

                    with torch.no_grad():
                        out1t = target(s1)
                        v1 = out1t["v"]
                        target_q = r2 + conf.gamma * (1.0 - done) * v1

                    q_a = q.gather(1, a.view(-1, 1)).squeeze(-1)

                    q_loss = F.smooth_l1_loss(q_a, target_q, beta=conf.huber_delta)

                    diff = (q_a.detach() - v)
                    v_loss = expectile_loss(diff, tau=conf.expectile_tau)

                    adv = (q_a.detach() - v.detach())
                    adv = adv / (adv.abs().mean().clamp_min(1e-6))
                    adv = torch.clamp(adv, -conf.adv_clip, conf.adv_clip)
                    w_adv = torch.exp(torch.clamp(adv / max(1e-6, conf.awr_temp), -conf.w_adv_clip, conf.w_adv_clip)).detach()

                    pi_loss = (sw * w_adv * bce).sum() / den

                    loss = conf.pi_coef * pi_loss + conf.q_coef * q_loss + conf.v_coef * v_loss

                    if conf.ent_coef > 0:
                        p_cli = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
                        ent = -(p_cli * torch.log(p_cli) + (1 - p_cli) * torch.log(1 - p_cli)).mean()
                        loss = loss - conf.ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema_update(target, model, ema=conf.ema)

                train_pi.append(float(pi_loss.item()))

            val_best = eval_val_dir_style(model, va_loader, mu_t, sd_t, thresholds)
            epoch_global = offset + ep

            print(f"[IQL-DIR:{name}] Ep {ep:02d} train_pi={np.mean(train_pi):.4f} | VAL F1={val_best['f1']:.3f} thr={val_best['thr']:.3f}")

            improved = val_best["f1"] > (best_f1 + conf.early_stop_min_delta)
            if improved:
                best_f1 = float(val_best["f1"])
                best_thr = float(val_best["thr"])
                best_epoch = int(epoch_global)
                epochs_since_improve = 0
                save_bundle(epoch_global, val_best)
                print(f"✅ Saved BEST by val_f1={best_f1:.4f} @thr={best_thr:.3f} -> {conf.out_ckpt}")
            else:
                epochs_since_improve += 1

            if ep >= conf.early_stop_warmup and epochs_since_improve >= conf.early_stop_patience:
                print(f"⛔ Early stopping. Best val_f1={best_f1:.4f} at epoch {best_epoch}.")
                return

    if conf.warmup_epochs > 0:
        run_phase("WARMUP", conf.warmup_epochs, "bce", opt_warm, 0)
    if conf.iql_epochs > 0:
        run_phase("IQL", conf.iql_epochs, "iql", opt_iql, conf.warmup_epochs)

    return {"out_ckpt": conf.out_ckpt, "best_f1": best_f1, "best_thr": best_thr, "best_epoch": best_epoch}


def run(conf: DirIQLConfig) -> Dict[str, Any]:
    return train_loop(conf)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser("IQL DIR Baseline (TRUE NONHOLD: CLD vs CLI)")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--out", default="checkpoints/baselines/dir_iql.pt")

    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--iql-epochs", type=int, default=20)

    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr-warmup", type=float, default=2e-4)
    p.add_argument("--lr-iql", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tau", type=float, default=0.7)
    p.add_argument("--awr-temp", type=float, default=2.0)
    p.add_argument("--adv-clip", type=float, default=2.0)
    p.add_argument("--w-adv-clip", type=float, default=0.7)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.995)

    p.add_argument("--q-coef", type=float, default=1.0)
    p.add_argument("--v-coef", type=float, default=1.0)
    p.add_argument("--pi-coef", type=float, default=1.0)
    p.add_argument("--ent", type=float, default=0.0)

    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=100.0)

    p.add_argument("--mix-cld", type=float, default=0.50)
    p.add_argument("--mix-cli", type=float, default=0.50)
    p.add_argument("--boost-cld", type=float, default=1.0)
    p.add_argument("--boost-cli", type=float, default=1.0)

    p.add_argument("--thr-min", type=float, default=0.05)
    p.add_argument("--thr-max", type=float, default=0.95)
    p.add_argument("--thr-steps", type=int, default=19)

    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--early-stop-warmup", type=int, default=2)

    args = p.parse_args(argv)

    conf = DirIQLConfig(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        out_ckpt=args.out,

        warmup_epochs=args.warmup_epochs,
        iql_epochs=args.iql_epochs,

        batch_size=args.batch_size,
        lr_warmup=args.lr_warmup,
        lr_iql=args.lr_iql,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        seed=args.seed,

        expectile_tau=args.tau,
        awr_temp=args.awr_temp,
        adv_clip=args.adv_clip,
        w_adv_clip=args.w_adv_clip,
        huber_delta=args.huber_delta,
        ema=args.ema,

        q_coef=args.q_coef,
        v_coef=args.v_coef,
        pi_coef=args.pi_coef,
        ent_coef=args.ent,

        reward_clip=args.reward_clip,
        reward_scale=args.reward_scale,

        mix_cld=args.mix_cld,
        mix_cli=args.mix_cli,
        boost_cld=args.boost_cld,
        boost_cli=args.boost_cli,

        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_steps=args.thr_steps,

        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup=args.early_stop_warmup,
    )

    run(conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
