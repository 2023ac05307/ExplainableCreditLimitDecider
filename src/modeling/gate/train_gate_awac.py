#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gate_awac.py
------------------
Repo-aligned Gate trainer (HOLD vs NONHOLD) using AWAC-style objectives.

Key properties:
- Config-driven: GateTrainConfig + run(conf) + main()
- Uses GateActorCritic / GateModelConfig (repo model)
- Option A checkpoint: ckpt["model"] stores full nn.Module for serving
- Also stores state_dict for portability and evaluation compatibility
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.modeling.gate.model import GateActorCritic, GateModelConfig

# ---------------------------------------------------------------------
# NOTE:
# The following helpers are assumed to exist in your repo already
# (they were referenced in your original script).
#
# Keep these imports pointing to YOUR actual locations.
# If your repo has these in a different module, adjust paths below.
# ---------------------------------------------------------------------
from src.modeling.data.datasets import TrajDatasetGATE  # must yield (s,a,r,s1,done,w)
from src.modeling.utils.seed import set_seed
from src.modeling.utils.sampler import build_weighted_sampler
from src.modeling.utils.ema import ema_update
from src.modeling.utils.eval_gate import eval_val_gate_style

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class GateTrainConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str = "checkpoints/gate_awac_best.pt"

    # epochs
    warmup_epochs: int = 3
    awac_epochs: int = 20

    # optimization
    batch_size: int = 4096
    lr_warmup: float = 2e-4
    lr_awac: float = 2e-4
    weight_decay: float = 1e-5
    gamma: float = 0.99
    seed: int = 42

    # AWAC knobs
    awr_temp: float = 1.2
    adv_clip: float = 3.0
    w_adv_clip: float = 1.5
    vf_coef: float = 0.01
    ent_coef: float = 5e-4
    ema: float = 0.995
    huber_delta: float = 1.0

    # reward shaping
    reward_clip: float = 10.0
    reward_scale: float = 1.0

    # sampling knobs (Gate labels: 0=HOLD, 1=NONHOLD)
    mix_hold: float = 0.50
    mix_nonhold: float = 0.50
    boost_hold: float = 1.0
    boost_nonhold: float = 1.0

    # threshold sweep
    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_steps: int = 37

    # early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    early_stop_warmup: int = 0


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_loop(conf: GateTrainConfig) -> Dict[str, Any]:
    """
    Repo-aligned Gate training loop.

    Checkpoint (Option A):
      - "model": full nn.Module (for src/serving/model_loader.py)
      - "state_dict": explicit weights (for portability/eval)
    """
    set_seed(conf.seed)

    # -----------------------------
    # Dataset
    # -----------------------------
    ds_tr = TrajDatasetGATE(conf.train_parquet, scaler=None)
    ds_va = TrajDatasetGATE(conf.val_parquet, scaler=None)  # IMPORTANT: no scaler here


    obs_dim = int(ds_tr.s.shape[1])
    print(f"TRAIN: {len(ds_tr):,} rows | obs_dim={obs_dim} | device={DEVICE}")
    print("TRAIN counts:", {0: int(ds_tr.action_counts[0]), 1: int(ds_tr.action_counts[1])})
    print(f"VAL:   {len(ds_va):,} rows")
    print("VAL counts:", {0: int(ds_va.action_counts[0]), 1: int(ds_va.action_counts[1])})
    print("Label map: 0=HOLD, 1=NONHOLD")

    sampler = build_weighted_sampler(
        ds_tr,
        mix0=conf.mix_hold,
        mix1=conf.mix_nonhold,
        boost0=conf.boost_hold,
        boost1=conf.boost_nonhold,
    )

    # train_loader = DataLoader(
    #     ds_tr,
    #     batch_size=conf.batch_size,
    #     sampler=sampler,
    #     drop_last=True,
    #     num_workers=0,
    #     pin_memory=(DEVICE == "cuda"),
    # )

    train_loader = DataLoader(
        ds_tr,
        batch_size=conf.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=4,                 # try 4, or 8
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,       # keeps workers alive
        prefetch_factor=2,
    )    

    # val_loader = DataLoader(
    #     ds_va,
    #     batch_size=conf.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    #     pin_memory=(DEVICE == "cuda"),
    # )

    val_loader = DataLoader(
        ds_va,
        batch_size=conf.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    thresholds = np.linspace(conf.thr_min, conf.thr_max, conf.thr_steps).astype(np.float32)

    # -----------------------------
    # Model (repo model)
    # -----------------------------
    model_cfg = GateModelConfig(obs_dim=obs_dim, dropout=0.10, depth=2, hidden=256, layer_norm=False)
    model = GateActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target = GateActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    mu_t = torch.as_tensor(ds_tr.mu, device=DEVICE, dtype=torch.float32)
    sd_t = torch.as_tensor(ds_tr.sd, device=DEVICE, dtype=torch.float32)

    os.makedirs(os.path.dirname(conf.out_ckpt) or ".", exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = 0
    best_thr = 0.5
    epochs_since_improve = 0

    # -----------------------------
    # Save checkpoint (Option A)
    # -----------------------------
    def save_bundle(epoch_global: int, val_best: Dict[str, Any]) -> None:
        bundle = {
            "task": "gate",
            "model": model,  # Option A: store full module for serving
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
            "label_map": {0: "HOLD", 1: "NONHOLD"},

            "config": conf.__dict__,
        }
        torch.save(bundle, conf.out_ckpt)

    def bce_loss_from_logits(logit: torch.Tensor, y_float: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logit, y_float, reduction="none")

    # -----------------------------
    # Phase runner
    # -----------------------------
    def run_phase(
        phase_name: str,
        epochs: int,
        mode: str,
        optimizer: torch.optim.Optimizer,
        epoch_offset: int,
    ) -> None:
        nonlocal best_val_f1, best_epoch, best_thr, epochs_since_improve

        epochs_since_improve = 0

        for ep in range(1, epochs + 1):
            model.train()
            train_pi_losses = []
            train_acc05 = []
            train_bce_losses = []      
            train_acc05 = []
            wadv_means = []            
            wadv_stds = []                         

            for i, (s_np, a_np, r_np, s1_np, done_np, w_np) in enumerate(train_loader):
                # s_raw = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32)
                # s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
                # a_long = torch.as_tensor(a_np, device=DEVICE, dtype=torch.long)
                 
                # r = torch.as_tensor(r_np, device=DEVICE, dtype=torch.float32)
                # done = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
                # sw = torch.as_tensor(w_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)
                
                if not torch.is_tensor(s_np):
                    s_raw = torch.from_numpy(s_np).float()
                    s1_raw = torch.from_numpy(s1_np).float()
                    a_long = torch.from_numpy(a_np).long()
                    r = torch.from_numpy(r_np).float()
                    done = torch.from_numpy(done_np).float()
                    sw = torch.from_numpy(w_np).float()
                else:
                    s_raw = s_np.float()
                    s1_raw = s1_np.float()
                    a_long = a_np.long()
                    r = r_np.float()
                    done = done_np.float()
                    sw = w_np.float()

                y = a_long.float().to(DEVICE, non_blocking=True)
                s_raw = s_raw.to(DEVICE, non_blocking=True)
                s1_raw = s1_raw.to(DEVICE, non_blocking=True)
                a_long = a_long.to(DEVICE, non_blocking=True)
                r = r.to(DEVICE, non_blocking=True)
                done = done.to(DEVICE, non_blocking=True)
                sw = sw.to(DEVICE, non_blocking=True)

                s = (s_raw - mu_t) / sd_t
                s1 = (s1_raw - mu_t) / sd_t

                out = model(s)
                logit = out["logit"]
                q = out.get("q", None)  # shape [B,2] if present
                v = out.get("v", None)  # shape [B] if present

                bce = bce_loss_from_logits(logit, y)
                den = sw.sum().clamp_min(1.0)

            # ---- DEBUG: sanity checks once (first epoch, first batch) ----
                if ep == 1 and i == 0:
                    y_cpu = y.detach().cpu().numpy()
                    print("[DEBUG] first batch class counts:",
                      {0: int((y_cpu == 0).sum()), 1: int((y_cpu == 1).sum())})

                    r_cpu = r.detach().cpu()
                    print("[DEBUG] reward r(raw):",
                      "min", float(r_cpu.min()), "max", float(r_cpu.max()),
                      "mean", float(r_cpu.mean()), "std", float(r_cpu.std()))
                    print("[DEBUG] reward_scale", float(conf.reward_scale),
                      "reward_clip", float(conf.reward_clip))

                if mode == "bce":
                    pi_loss = (sw * bce).sum() / den
                    loss = pi_loss

                else:
                    # reward scaling/clipping
                    r2 = r / max(1e-6, conf.reward_scale)
                    r2 = torch.clamp(r2, -conf.reward_clip, conf.reward_clip)

                    # ---- DEBUG: scaled reward once (first epoch, first batch, AWAC only) ----
                    if ep == 1 and i == 0:
                        r2_cpu = r2.detach().cpu()
                        print("[DEBUG] reward r2(scaled+clipped):",
                            "min", float(r2_cpu.min()), "max", float(r2_cpu.max()),
                            "mean", float(r2_cpu.mean()), "std", float(r2_cpu.std()))
                    # ----------------------------------------------------------------------

                    # target Q
                    with torch.no_grad():
                        out1t = target(s1)
                        q1_t = out1t["q"]
                        max_q1 = q1_t.max(dim=-1).values
                        target_q = r2 + conf.gamma * (1.0 - done) * max_q1

                    q_a = q.gather(1, a_long.view(-1, 1)).squeeze(-1)

                    adv = (q_a - v).detach()
                    adv = adv / (adv.abs().mean().clamp_min(1e-6))
                    adv = torch.clamp(adv, -conf.adv_clip, conf.adv_clip)

                    w_adv = torch.exp(
                        torch.clamp(adv / max(1e-6, conf.awr_temp), -conf.w_adv_clip, conf.w_adv_clip)
                    ).detach()

                    # ---- DEBUG: AWAC weight stats (shows whether AWAC is actually active) ----
                    if ep == 1 and i == 0:
                        w_cpu = w_adv.detach().cpu()
                        print("[DEBUG] w_adv stats:",
                            "mean", float(w_cpu.mean()), "std", float(w_cpu.std()),
                            "min", float(w_cpu.min()), "max", float(w_cpu.max()))
                    # -----------------------------------------------------------------------

                    pi_loss = (sw * w_adv * bce).sum() / den

                    td = F.smooth_l1_loss(q_a, target_q, beta=conf.huber_delta, reduction="none")
                    q_td = (sw * td).sum() / den

                    p1 = torch.sigmoid(logit)
                    pi2 = torch.stack([1.0 - p1, p1], dim=-1)
                    v_target = (pi2 * q).sum(dim=-1).detach()
                    vr = F.smooth_l1_loss(v, v_target, beta=conf.huber_delta, reduction="none")
                    v_reg = (sw * vr).sum() / den

                    critic_loss = q_td + 0.5 * v_reg
                    loss = pi_loss + conf.vf_coef * critic_loss

                    if conf.ent_coef > 0:
                        ent = -(
                            p1 * torch.log(p1 + 1e-8) + (1.0 - p1) * torch.log(1.0 - p1 + 1e-8)
                        ).mean()
                        loss = loss - conf.ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema_update(target, model, ema=conf.ema)

                with torch.no_grad():
                    p = torch.sigmoid(logit)
                    pred = (p >= 0.5).float()
                    acc05 = (pred == y).float().mean().item()

                train_pi_losses.append(float(pi_loss.item()))
                train_acc05.append(float(acc05))

            val_best = eval_val_gate_style(model, val_loader, mu_t, sd_t, thresholds)
            epoch_global = epoch_offset + ep

            # build log line (add AWAC-only diagnostics)
            log_line = (
                f"[{phase_name}] Ep {ep:02d} | "
                f"train_pi={np.mean(train_pi_losses):.4f} train_acc@0.5={np.mean(train_acc05):.3f} | "
            )
            if mode != "bce" and len(train_bce_losses) > 0:
                log_line += (
                    f"train_bce={np.mean(train_bce_losses):.4f} "
                    f"w_adv(mean/std)={np.mean(wadv_means):.3f}/{np.mean(wadv_stds):.3f} | "
                )

            log_line += (
                f"VAL thr={val_best['thr']:.3f} acc={val_best['acc']:.3f} "
                f"prec={val_best['prec']:.3f} rec={val_best['rec']:.3f} "
                f"F1={val_best['f1']:.3f} bal_acc={val_best['bal_acc']:.3f} "
                f"(tp={val_best['tp']} fp={val_best['fp']} fn={val_best['fn']} tn={val_best['tn']})"
            )
            print(log_line)

            improved = (val_best["f1"] > (best_val_f1 + conf.early_stop_min_delta))
            if improved:
                best_val_f1 = float(val_best["f1"])
                best_thr = float(val_best["thr"])
                best_epoch = int(epoch_global)
                epochs_since_improve = 0

                save_bundle(epoch_global, val_best)
                print(f"Saved BEST checkpoint by val_f1={best_val_f1:.4f} @thr={best_thr:.3f} -> {conf.out_ckpt}")
            else:
                epochs_since_improve += 1

            if ep >= conf.early_stop_warmup and epochs_since_improve >= conf.early_stop_patience:
                print(f"Early stopping. Best val_f1={best_val_f1:.4f} at epoch {best_epoch}.")
                return

    # -----------------------------
    # Run phases
    # -----------------------------
    opt_warm = torch.optim.AdamW(model.parameters(), lr=conf.lr_warmup, weight_decay=conf.weight_decay)
    if conf.warmup_epochs > 0:
        run_phase("WARMUP", conf.warmup_epochs, mode="bce", optimizer=opt_warm, epoch_offset=0)

    opt_awac = torch.optim.AdamW(model.parameters(), lr=conf.lr_awac, weight_decay=conf.weight_decay)
    if conf.awac_epochs > 0:
        run_phase("AWAC", conf.awac_epochs, mode="awac", optimizer=opt_awac, epoch_offset=conf.warmup_epochs)

    print("Done. Best val_f1:", best_val_f1, "best_thr:", best_thr, "best_epoch:", best_epoch)

    return {
        "out_ckpt": conf.out_ckpt,
        "best_val_f1": float(best_val_f1),
        "best_thr": float(best_thr),
        "best_epoch": int(best_epoch),
    }


def run(conf: GateTrainConfig) -> Dict[str, Any]:
    return train_loop(conf)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser("Gate AWAC Trainer (HOLD vs NONHOLD)")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--out", default="checkpoints/gate_awac_best.pt")

    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--awac-epochs", type=int, default=20)

    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr-warmup", type=float, default=2e-4)
    p.add_argument("--lr-awac", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--awr-temp", type=float, default=1.2)
    p.add_argument("--adv-clip", type=float, default=3.0)
    p.add_argument("--w-adv-clip", type=float, default=1.5)
    p.add_argument("--vf-coef", type=float, default=0.01)
    p.add_argument("--ent", type=float, default=5e-4)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--huber-delta", type=float, default=1.0)

    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=100.0)

    p.add_argument("--mix-hold", type=float, default=0.50)
    p.add_argument("--mix-nonhold", type=float, default=0.50)
    p.add_argument("--boost-hold", type=float, default=1.0)
    p.add_argument("--boost-nonhold", type=float, default=1.0)

    p.add_argument("--thr-min", type=float, default=0.05)
    p.add_argument("--thr-max", type=float, default=0.95)
    p.add_argument("--thr-steps", type=int, default=37)

    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--early-stop-warmup", type=int, default=0)

    args = p.parse_args(argv)

    conf = GateTrainConfig(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        out_ckpt=args.out,

        warmup_epochs=args.warmup_epochs,
        awac_epochs=args.awac_epochs,

        batch_size=args.batch_size,
        lr_warmup=args.lr_warmup,
        lr_awac=args.lr_awac,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        seed=args.seed,

        awr_temp=args.awr_temp,
        adv_clip=args.adv_clip,
        w_adv_clip=args.w_adv_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent,
        ema=args.ema,
        huber_delta=args.huber_delta,

        reward_clip=args.reward_clip,
        reward_scale=args.reward_scale,

        mix_hold=args.mix_hold,
        mix_nonhold=args.mix_nonhold,
        boost_hold=args.boost_hold,
        boost_nonhold=args.boost_nonhold,

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
