#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.modeling.direction.model import DirActorCritic, DirModelConfig
from src.modeling.data.datasets import TrajDatasetDIR  # must yield (s,a,r,s1,done,w)
from src.modeling.utils.seed import set_seed
from src.modeling.utils.ema import ema_update

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# ✅ Midsem-exact sampler behavior
# -----------------------------
def build_weighted_sampler_midsem(
    ds: TrajDatasetDIR,
    mix0: float,
    mix1: float,
    boost0: float,
    boost1: float,
) -> WeightedRandomSampler:
    """
    Midsem behavior:
      - sampler MUST NOT multiply ds.w (sample_weight)
      - exposure controlled only by mix/boost/counts
    """
    mix = np.array([mix0, mix1], dtype=np.float32)
    mix = mix / mix.sum()
    boost = np.array([boost0, boost1], dtype=np.float32)

    # ds.action_counts could be dict or array depending on your dataset implementation
    if isinstance(ds.action_counts, dict):
        counts = np.array([ds.action_counts.get(0, 0), ds.action_counts.get(1, 0)], dtype=np.float64)
    else:
        counts = np.asarray(ds.action_counts, dtype=np.float64)

    counts = np.maximum(counts, 1.0)
    class_base = (mix.astype(np.float64) / counts) * boost.astype(np.float64)

    a = np.asarray(ds.a, dtype=np.int64)
    samp_w = np.zeros(len(ds), dtype=np.float64)
    samp_w[a == 0] = class_base[0]
    samp_w[a == 1] = class_base[1]
    samp_w = np.clip(samp_w, 1e-12, None)

    return WeightedRandomSampler(
        weights=torch.as_tensor(samp_w),
        num_samples=len(ds),
        replacement=True,
    )


# -----------------------------
# Metrics (same as midsem)
# -----------------------------
@torch.no_grad()
def sweep_thresholds(y_true_np: np.ndarray, p_np: np.ndarray, thresholds: np.ndarray) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for thr in thresholds:
        pred = (p_np >= thr).astype(np.int64)

        tp = int(((pred == 1) & (y_true_np == 1)).sum())
        fp = int(((pred == 1) & (y_true_np == 0)).sum())
        fn = int(((pred == 0) & (y_true_np == 1)).sum())
        tn = int(((pred == 0) & (y_true_np == 0)).sum())

        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))
        f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

        tpr = rec
        tnr = tn / max(1, (tn + fp))
        bal_acc = 0.5 * (tpr + tnr)

        cand = dict(
            thr=float(thr), acc=float(acc), prec=float(prec), rec=float(rec),
            f1=float(f1), bal_acc=float(bal_acc),
            tp=tp, fp=fp, fn=fn, tn=tn
        )
        if (best is None) or (cand["f1"] > best["f1"]) or (cand["f1"] == best["f1"] and cand["bal_acc"] > best["bal_acc"]):
            best = cand
    return best if best is not None else dict(thr=0.5, acc=0.0, prec=0.0, rec=0.0, f1=0.0, bal_acc=0.0, tp=0, fp=0, fn=0, tn=0)


@torch.no_grad()
def eval_val_dir(model: nn.Module, loader: DataLoader, mu_t: torch.Tensor, sd_t: torch.Tensor, thresholds: np.ndarray) -> Dict[str, Any]:
    model.eval()
    ys, ps = [], []

    for s, a, r, s1, done, w in loader:
        if not torch.is_tensor(s):
            s = torch.as_tensor(s, dtype=torch.float32)
        s = s.to(DEVICE, non_blocking=True)
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        sz = (s - mu_t) / sd_t

        logit = model(sz)["logit"]
        p_cli = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float32)  # ✅ P(CLI)

        ys.append(np.asarray(a, dtype=np.int64))
        ps.append(p_cli)

    y_true = np.concatenate(ys, axis=0)
    p_all  = np.concatenate(ps, axis=0)
    return sweep_thresholds(y_true, p_all, thresholds)


@dataclass
class DirTrainConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str = "checkpoints/dir_awac_best.pt"

    warmup_epochs: int = 5
    awac_epochs: int = 20

    batch_size: int = 4096
    lr_warmup: float = 2e-4
    lr_awac: float = 1e-4
    weight_decay: float = 1e-5

    gamma: float = 0.99
    seed: int = 42

    # ✅ midsem-parity defaults
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.10
    layer_norm: bool = False

    # AWAC knobs
    awr_temp: float = 2.0
    adv_clip: float = 2.0
    w_adv_clip: float = 0.7
    vf_coef: float = 0.01
    ent_coef: float = 0.0
    ema: float = 0.995
    huber_delta: float = 1.0

    reward_clip: float = 10.0
    reward_scale: float = 100.0

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


def train_loop(conf: DirTrainConfig) -> Dict[str, Any]:
    set_seed(conf.seed)

    ds_tr = TrajDatasetDIR(conf.train_parquet, scaler=None)
    ds_va = TrajDatasetDIR(conf.val_parquet, scaler=None)


    obs_dim = int(ds_tr.s.shape[1])
    print(f"TRAIN: {len(ds_tr):,} rows | obs_dim={obs_dim} | device={DEVICE}")
    print(f"State cols: {len(ds_tr.state_cols)} | Next cols: {len(ds_tr.next_cols)}")
    print("TRAIN counts:", {0: int(ds_tr.action_counts[0] if not isinstance(ds_tr.action_counts, dict) else ds_tr.action_counts.get(0, 0)),
                           1: int(ds_tr.action_counts[1] if not isinstance(ds_tr.action_counts, dict) else ds_tr.action_counts.get(1, 0))})
    print(f"VAL:   {len(ds_va):,} rows")
    print("VAL counts:", {0: int(ds_va.action_counts[0] if not isinstance(ds_va.action_counts, dict) else ds_va.action_counts.get(0, 0)),
                         1: int(ds_va.action_counts[1] if not isinstance(ds_va.action_counts, dict) else ds_va.action_counts.get(1, 0))})
    print("Label map: 0=CLD, 1=CLI")

    sampler = build_weighted_sampler_midsem(ds_tr, conf.mix_cld, conf.mix_cli, conf.boost_cld, conf.boost_cli)

    train_loader = DataLoader(
        ds_tr,
        batch_size=conf.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

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

    model_cfg = DirModelConfig(
        obs_dim=obs_dim, hidden=conf.hidden, depth=conf.depth,
        dropout=conf.dropout, layer_norm=conf.layer_norm
    )
    model = DirActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target = DirActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
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

    def save_bundle(epoch_global: int, val_best: Dict[str, Any]) -> None:
        bundle = {
            "task": "dir",
            "model": model,                         # serving convenience
            "state_dict": model.state_dict(),        # portability
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
        }
        torch.save(bundle, conf.out_ckpt)

    def bce_loss_from_logits(logit: torch.Tensor, y_float: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logit, y_float, reduction="none")

    def run_phase(phase_name: str, epochs: int, mode: str, optimizer: torch.optim.Optimizer, epoch_offset: int) -> None:
        nonlocal best_val_f1, best_epoch, best_thr, epochs_since_improve

        for ep in range(1, epochs + 1):
            model.train()
            train_pi_losses = []
            train_acc05 = []

            for s, a, r, s1, done, w in train_loader:

                if not torch.is_tensor(s):
                    s_raw  = torch.from_numpy(s).float()
                    s1_raw = torch.from_numpy(s1).float()
                    a_long = torch.from_numpy(a).long()
                    r      = torch.from_numpy(r).float()
                    done   = torch.from_numpy(done).float()
                    sw     = torch.from_numpy(w).float()
                else:
                    s_raw, s1_raw = s.float(), s1.float()
                    a_long = a.long()
                    r, done, sw = r.float(), done.float(), w.float()

                # non_blocking transfers (requires pin_memory=True)
                s_raw  = s_raw.to(DEVICE, non_blocking=True)
                s1_raw = s1_raw.to(DEVICE, non_blocking=True)
                a_long = a_long.to(DEVICE, non_blocking=True)
                r      = r.to(DEVICE, non_blocking=True)
                done   = done.to(DEVICE, non_blocking=True)
                sw     = sw.to(DEVICE, non_blocking=True)
                y = a_long.float()

                s  = (s_raw  - mu_t) / sd_t
                s1 = (s1_raw - mu_t) / sd_t

                out = model(s)
                logit = out["logit"]
                q = out["q"]
                v = out["v"]

                bce = bce_loss_from_logits(logit, y)
                den = sw.sum().clamp_min(1.0)

                if mode == "bce":
                    pi_loss = (sw * bce).sum() / den
                    loss = pi_loss
                else:
                    r2 = (r / max(1e-6, conf.reward_scale))
                    r2 = torch.clamp(r2, -conf.reward_clip, conf.reward_clip)

                    with torch.no_grad():
                        out1t = target(s1)
                        max_q1 = out1t["q"].max(dim=-1).values
                        target_q = r2 + conf.gamma * (1.0 - done) * max_q1

                    q_a = q.gather(1, a_long.view(-1, 1)).squeeze(-1)

                    adv = (q_a - v).detach()
                    adv = adv / (adv.abs().mean().clamp_min(1e-6))
                    adv = torch.clamp(adv, -conf.adv_clip, conf.adv_clip)

                    w_adv = torch.exp(
                        torch.clamp(adv / max(1e-6, conf.awr_temp), -conf.w_adv_clip, conf.w_adv_clip)
                    ).detach()

                    pi_loss = (sw * w_adv * bce).sum() / den

                    q_td = F.smooth_l1_loss(q_a, target_q, beta=conf.huber_delta)

                    # ✅ p_cli is ALWAYS sigmoid(logit)
                    p_cli = torch.sigmoid(logit)
                    pi2 = torch.stack([1.0 - p_cli, p_cli], dim=-1)

                    v_target = (pi2 * q).sum(dim=-1).detach()
                    v_reg = F.smooth_l1_loss(v, v_target, beta=conf.huber_delta)

                    critic_loss = q_td + 0.5 * v_reg
                    loss = pi_loss + conf.vf_coef * critic_loss

                    if conf.ent_coef > 0:
                        ent = -(p_cli * torch.log(p_cli + 1e-8) + (1.0 - p_cli) * torch.log(1.0 - p_cli + 1e-8)).mean()
                        loss = loss - conf.ent_coef * ent

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema_update(target, model, ema=conf.ema)

                with torch.no_grad():
                    p_cli = torch.sigmoid(logit)
                    pred = (p_cli >= 0.5).float()
                    acc05 = (pred == y).float().mean().item()

                train_pi_losses.append(float(pi_loss.item()))
                train_acc05.append(float(acc05))

            val_best = eval_val_dir(model, val_loader, mu_t, sd_t, thresholds)
            epoch_global = epoch_offset + ep

            print(
                f"[{phase_name}] Ep {ep:02d} | "
                f"train_pi={np.mean(train_pi_losses):.4f} train_acc@0.5={np.mean(train_acc05):.3f} | "
                f"VAL thr={val_best['thr']:.3f} acc={val_best['acc']:.3f} "
                f"prec={val_best['prec']:.3f} rec={val_best['rec']:.3f} "
                f"F1={val_best['f1']:.3f} bal_acc={val_best['bal_acc']:.3f} "
                f"(tp={val_best['tp']} fp={val_best['fp']} fn={val_best['fn']} tn={val_best['tn']})"
            )

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

    opt_warm = torch.optim.AdamW(model.parameters(), lr=conf.lr_warmup, weight_decay=conf.weight_decay)
    if conf.warmup_epochs > 0:
        run_phase("WARMUP", conf.warmup_epochs, mode="bce", optimizer=opt_warm, epoch_offset=0)

    opt_awac = torch.optim.AdamW(model.parameters(), lr=conf.lr_awac, weight_decay=conf.weight_decay)
    if conf.awac_epochs > 0:
        run_phase("AWAC", conf.awac_epochs, mode="awac", optimizer=opt_awac, epoch_offset=conf.warmup_epochs)

    print("Done. Best val_f1:", best_val_f1, "best_thr:", best_thr, "best_epoch:", best_epoch)
    return {"best_val_f1": float(best_val_f1), "best_thr": float(best_thr), "best_epoch": int(best_epoch), "out_ckpt": conf.out_ckpt}


def run(conf: DirTrainConfig) -> Dict[str, Any]:
    return train_loop(conf)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser("DIR AWAC Trainer (midsem-parity)")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--out", default="checkpoints/dir_awac_best.pt")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--awac-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr-warmup", type=float, default=2e-4)
    p.add_argument("--lr-awac", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--layer-norm", action="store_true", default=False)
    args = p.parse_args(argv)

    conf = DirTrainConfig(
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
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
    )
    run(conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
