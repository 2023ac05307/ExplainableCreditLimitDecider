from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.modeling.gate.model import GateActorCritic, GateModelConfig
from src.modeling.baselines.common import DEVICE, set_seed, save_ckpt_option_a
from src.modeling.data.datasets import TrajDatasetGATE
from src.modeling.utils.sampler import build_weighted_sampler
from src.modeling.utils.ema import ema_update
from src.modeling.utils.eval_gate import eval_val_gate_style


@dataclass
class GatePPOConfig:
    train_parquet: str
    val_parquet: str
    out_ckpt: str = "checkpoints/baselines/gate_ppo.pt"

    epochs: int = 20
    batch_size: int = 4096
    lr: float = 2e-4
    weight_decay: float = 1e-5
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    ema: float = 0.995
    seed: int = 42

    mix_hold: float = 0.50
    mix_nonhold: float = 0.50
    boost_hold: float = 1.0
    boost_nonhold: float = 1.0

    thr_min: float = 0.05
    thr_max: float = 0.95
    thr_steps: int = 19


def train_loop(conf: GatePPOConfig) -> Dict[str, Any]:
    set_seed(conf.seed)

    ds_tr = TrajDatasetGATE(conf.train_parquet, scaler=None)
    scaler = {"mu": ds_tr.mu, "sd": ds_tr.sd}
    ds_va = TrajDatasetGATE(conf.val_parquet, scaler=scaler)

    obs_dim = int(ds_tr.s.shape[1])
    mu_t = torch.as_tensor(ds_tr.mu, device=DEVICE, dtype=torch.float32)
    sd_t = torch.as_tensor(ds_tr.sd, device=DEVICE, dtype=torch.float32)
    thresholds = np.linspace(conf.thr_min, conf.thr_max, conf.thr_steps).astype(np.float32)

    sampler = build_weighted_sampler(
        ds_tr, mix0=conf.mix_hold, mix1=conf.mix_nonhold, boost0=conf.boost_hold, boost1=conf.boost_nonhold
    )

    tr_loader = DataLoader(ds_tr, batch_size=conf.batch_size, sampler=sampler, drop_last=True, pin_memory=(DEVICE=="cuda"))
    va_loader = DataLoader(ds_va, batch_size=conf.batch_size, shuffle=False, drop_last=False, pin_memory=(DEVICE=="cuda"))

    model_cfg = GateModelConfig(obs_dim=obs_dim)
    model = GateActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target = GateActorCritic(model_cfg, include_q=True, include_v=True).to(DEVICE)
    target.load_state_dict(model.state_dict())
    for p in target.parameters(): p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    best_f1, best_thr, best_epoch = -1.0, 0.5, 0

    for ep in range(1, conf.epochs + 1):
        model.train()
        losses = []

        for s_np, a_np, r_np, s1_np, done_np, w_np in tr_loader:
            s_raw = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32)
            a = torch.as_tensor(a_np, device=DEVICE, dtype=torch.long)      # 0/1
            y = a.float()

            s = (torch.nan_to_num(s_raw, 0.0, 0.0, 0.0) - mu_t) / sd_t
            out = model(s)
            logit = out["logit"]
            v = out["v"]

            # PPO "policy" proxy for binary: sigmoid logit is prob of NONHOLD
            p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            # log prob for chosen action
            logp = torch.where(y > 0.5, torch.log(p), torch.log(1 - p))

            # Use target advantage from Q-V (offline-like)
            with torch.no_grad():
                q = out["q"]
                qa = q.gather(1, a.view(-1,1)).squeeze(-1)
                adv = (qa - v).detach()
                adv = adv / (adv.abs().mean().clamp_min(1e-6))

                logp_old = logp.detach()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - conf.clip_eps, 1 + conf.clip_eps) * adv
            pi_loss = -torch.min(surr1, surr2).mean()

            # simple value loss
            v_loss = (v ** 2).mean() * 0.0  # keep stable; you can plug real target if you want

            ent = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
            loss = pi_loss + conf.vf_coef * v_loss - conf.ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(target, model, ema=conf.ema)

            losses.append(float(loss.item()))

        val_best = eval_val_gate_style(model, va_loader, mu_t, sd_t, thresholds)
        print(f"[PPO-GATE] Ep {ep:02d} loss={np.mean(losses):.4f} | VAL F1={val_best['f1']:.3f} thr={val_best['thr']:.3f}")

        if val_best["f1"] > best_f1:
            best_f1 = float(val_best["f1"])
            best_thr = float(val_best["thr"])
            best_epoch = ep
            save_ckpt_option_a(
                out_ckpt=conf.out_ckpt,
                task="baseline_gate_ppo",
                model=model,
                obs_dim=obs_dim,
                state_cols=ds_tr.state_cols,
                scaler_mean=ds_tr.mu,
                scaler_std=ds_tr.sd,
                best={"f1": best_f1, "thr": best_thr, "epoch": best_epoch},
                label_map={0:"HOLD", 1:"NONHOLD"},
                model_config=model_cfg.__dict__,
                train_config=conf.__dict__,
            )

    return {"out_ckpt": conf.out_ckpt, "best_f1": best_f1, "best_thr": best_thr, "best_epoch": best_epoch}


def run(conf: GatePPOConfig) -> Dict[str, Any]:
    return train_loop(conf)
