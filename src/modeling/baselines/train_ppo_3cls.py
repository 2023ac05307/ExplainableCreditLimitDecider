# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# train_offline_ppo_3cls.py
# -------------------------
# Offline PPO for 3 discrete actions (0=HOLD, 1=CLI, 2=CLD) using logged transitions.

# Key idea (since dataset has no behavior log-probs):
# 1) Warmup: train a behavior cloning policy pi_b(a|s) on logged actions.
# 2) Freeze pi_b and run PPO updates on pi_theta using ratio exp(logp_new - logp_bc).
# 3) Critic is V(s) with EMA target V_t for TD bootstrap.
# 4) Action masking (guardrails) is applied to BOTH behavior and current policy logits.

# Input Parquet/CSV:
# - state features: columns starting with "s_"
# - next state     : columns starting with "s1_"
# - required cols  : action_id, reward, done
# - optional       : sample_weight

# Example:
# python train_offline_ppo_3cls.py \
#   --train-csv rl_dataset/merged/merged_train.csv \
#   --val-csv   rl_dataset/merged/merged_val.csv \
#   --epochs 25 --bc-epochs 3 --batch-size 4096 --lr 3e-4 \
#   --gamma 0.99 --clip-eps 0.2 --vf-coef 0.5 --ent 0.002 \
#   --mix-hold 0.50 --mix-cli 0.25 --mix-cld 0.25 \
#   --out checkpoints/offline_ppo_3cls.pt
# """

# import os, random, json
# import numpy as np
# import pandas as pd

from pathlib import Path
import glob


def read_table(path: str, low_memory: bool = False) -> pd.DataFrame:
    """Read a dataset from Parquet/CSV.

    Supports:
      - single file: .parquet/.pq or .csv
      - directory: reads all *.parquet/*.pq (preferred) else *.csv inside the directory

    Note: remaining training logic assumes a fully materialized pandas DataFrame.
    """
    p = Path(path)
    if p.is_dir():
        # Prefer parquet shards if present
        files = sorted(glob.glob(str(p / "*.parquet")) + glob.glob(str(p / "*.pq")))
        if not files:
            files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No parquet/csv files found in directory: {path}")

        dfs = []
        for fp in files:
            ext = Path(fp).suffix.lower()
            if ext in [".parquet", ".pq"]:
                dfs.append(pd.read_parquet(fp))
            elif ext == ".csv":
                dfs.append(pd.read_csv(fp, low_memory=low_memory))
            else:
                continue
        if not dfs:
            raise FileNotFoundError(f"No readable parquet/csv files found in directory: {path}")
        return pd.concat(dfs, axis=0, ignore_index=True)

    # Single file
    ext = p.suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    if ext == ".csv":
        return pd.read_csv(p, low_memory=low_memory)

    raise ValueError(f"Unsupported input type: {path} (expected .parquet/.pq or .csv, or a directory)")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ACTION_HOLD, ACTION_CLI, ACTION_CLD = 0, 1, 2


# # -----------------------------
# # Repro
# # -----------------------------
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def ema_update(target: nn.Module, source: nn.Module, ema: float):
#     with torch.no_grad():
#         for tp, sp in zip(target.parameters(), source.parameters()):
#             tp.data.mul_(ema).add_(sp.data, alpha=(1.0 - ema))


# def safe_mean(xs):
#     return float(np.mean(xs)) if len(xs) else 0.0


# # -----------------------------
# # Dataset
# # -----------------------------
# class TrajDataset(Dataset):
#     def __init__(self, csv_path: str, state_prefix="s_", next_prefix="s1_", scaler=None):
#         df = read_table(csv_path, low_memory=False).replace([np.inf, -np.inf], np.nan)

#         self.state_cols = [c for c in df.columns if c.startswith(state_prefix)]
#         self.next_cols  = [c for c in df.columns if c.startswith(next_prefix)]
#         if not self.state_cols:
#             raise RuntimeError("No state columns found (expected prefix 's_').")
#         if not self.next_cols:
#             raise RuntimeError("No next-state columns found (expected prefix 's1_').")

#         required = ["reward", "done", "action_id"]
#         for c in required:
#             if c not in df.columns:
#                 raise RuntimeError(f"Missing required column: {c}")

#         if "sample_weight" not in df.columns:
#             df["sample_weight"] = 1.0

#         # fill
#         df[self.state_cols] = df[self.state_cols].fillna(0.0)
#         df[self.next_cols]  = df[self.next_cols].fillna(0.0)
#         for c in required + ["sample_weight"]:
#             df[c] = df[c].fillna(0.0)

#         self.s   = df[self.state_cols].astype(np.float32).to_numpy(copy=True)
#         self.s1  = df[self.next_cols].astype(np.float32).to_numpy(copy=True)
#         self.a   = df["action_id"].astype(np.int64).to_numpy(copy=True)
#         self.r   = df["reward"].astype(np.float32).to_numpy(copy=True)
#         self.d   = df["done"].astype(np.float32).clip(0.0, 1.0).to_numpy(copy=True)
#         self.w   = df["sample_weight"].astype(np.float32).clip(0.0, 10.0).to_numpy(copy=True)

#         self.action_counts = np.array([(self.a == 0).sum(), (self.a == 1).sum(), (self.a == 2).sum()], dtype=np.int64)

#         # scaler (fit only on train, reuse on val)
#         if scaler is None:
#             mu = np.mean(self.s, axis=0).astype(np.float32)
#             sd = np.std(self.s, axis=0).astype(np.float32)
#             sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
#             self.mu, self.sd = mu, sd
#         else:
#             self.mu = scaler["mu"].astype(np.float32)
#             self.sd = scaler["sd"].astype(np.float32)

#     def __len__(self):
#         return int(self.a.shape[0])

#     def __getitem__(self, idx: int):
#         return self.s[idx], self.a[idx], self.r[idx], self.s1[idx], self.d[idx], self.w[idx]


# # -----------------------------
# # Model
# # -----------------------------
# class PPOActorCritic(nn.Module):
#     """
#     Actor: logits for Categorical over 3 actions.
#     Critic: V(s) scalar.
#     """
#     def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.05):
#         super().__init__()
#         self.backbone = nn.Sequential(
#             nn.Linear(obs_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#         )
#         self.pi = nn.Linear(hidden, 3)
#         self.v  = nn.Linear(hidden, 1)

#     def forward(self, obs: torch.Tensor):
#         x = self.backbone(obs)
#         logits = self.pi(x)             # [B,3]
#         v = self.v(x).squeeze(-1)       # [B]
#         return logits, v


# # -----------------------------
# # Guardrail Mask (RAW features)
# # -----------------------------
# def compute_action_mask(
#     s_raw: torch.Tensor,
#     state_cols: list,
#     min_score_for_cli: float = 680.0,
#     max_dpd_for_cli: float = 1.0,
#     min_pay_ratio_for_cli: float = 0.90,
#     min_util_for_cli: float = 0.25,
#     max_util_for_cli: float = 0.95,
#     min_overlimit_for_cld: float = 0.05,
#     min_dpd_for_cld: float = 2.0,
#     max_score_for_cld: float = 650.0
# ):
#     """
#     mask: [B,3] bool True=allowed.
#     If required columns not present => allow all.
#     """
#     B = s_raw.shape[0]
#     mask = torch.ones((B, 3), dtype=torch.bool, device=s_raw.device)

#     def col(name):
#         if name in state_cols:
#             return s_raw[:, state_cols.index(name)]
#         return None

#     score = col("s_external_score")
#     dpd   = col("s_dpd_count_12m")
#     pay   = col("s_payment_ratio")
#     utilm = col("s_max_utilization_6m")
#     over  = col("s_overlimit_rate_90d")

#     if score is None or dpd is None or pay is None or utilm is None:
#         return mask

#     cli_ok = (
#         (score >= min_score_for_cli) &
#         (dpd <= max_dpd_for_cli) &
#         (pay >= min_pay_ratio_for_cli) &
#         (utilm >= min_util_for_cli) &
#         (utilm <= max_util_for_cli)
#     )

#     if over is None:
#         cld_ok = (dpd >= min_dpd_for_cld) | (score <= max_score_for_cld)
#     else:
#         cld_ok = (dpd >= min_dpd_for_cld) | (over >= min_overlimit_for_cld) | (score <= max_score_for_cld)

#     mask[:, ACTION_HOLD] = True
#     mask[:, ACTION_CLI]  = cli_ok
#     mask[:, ACTION_CLD]  = cld_ok
#     return mask


# # -----------------------------
# # Sampler
# # -----------------------------
# def build_weighted_sampler(ds: TrajDataset, mix_hold, mix_cli, mix_cld, boost_hold, boost_cli, boost_cld):
#     mix = np.array([mix_hold, mix_cli, mix_cld], dtype=np.float32)
#     mix = mix / mix.sum()
#     boost = np.array([boost_hold, boost_cli, boost_cld], dtype=np.float32)

#     counts = ds.action_counts.astype(np.float64)
#     counts = np.maximum(counts, 1.0)

#     class_base = (mix.astype(np.float64) / counts) * boost.astype(np.float64)

#     samp_w = np.zeros(len(ds), dtype=np.float64)
#     samp_w[ds.a == ACTION_HOLD] = class_base[ACTION_HOLD]
#     samp_w[ds.a == ACTION_CLI]  = class_base[ACTION_CLI]
#     samp_w[ds.a == ACTION_CLD]  = class_base[ACTION_CLD]

#     # include row weights
#     samp_w = samp_w * ds.w.astype(np.float64)
#     samp_w = np.clip(samp_w, 1e-12, None)

#     sampler = WeightedRandomSampler(
#         weights=torch.as_tensor(samp_w),
#         num_samples=len(ds),
#         replacement=True,
#     )
#     return sampler, mix, boost


# # -----------------------------
# # Eval
# # -----------------------------
# @torch.no_grad()
# def eval_epoch(model, behavior, target, loader, mu_t, sd_t, state_cols,
#                gamma, reward_clip, reward_scale, vf_coef, ent_coef, clip_eps):
#     model.eval()
#     stats = {k: [] for k in ["loss","actor","vf","ent","approx_kl","clipfrac"]}

#     for s_np, a_np, r_np, s1_np, done_np, w_np in loader:
#         s_raw  = torch.as_tensor(s_np,  device=DEVICE, dtype=torch.float32)
#         s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
#         a      = torch.as_tensor(a_np,  device=DEVICE, dtype=torch.long)
#         r      = torch.as_tensor(r_np,  device=DEVICE, dtype=torch.float32)
#         done   = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
#         sw     = torch.as_tensor(w_np,  device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

#         s_raw  = torch.nan_to_num(s_raw,  nan=0.0, posinf=0.0, neginf=0.0)
#         s1_raw = torch.nan_to_num(s1_raw, nan=0.0, posinf=0.0, neginf=0.0)
#         r      = torch.nan_to_num(r,      nan=0.0, posinf=0.0, neginf=0.0)

#         mask = compute_action_mask(s_raw, state_cols)
#         a_allowed = mask.gather(1, a.view(-1, 1)).squeeze(1)
#         sw_eff = sw * a_allowed.float()

#         s  = (s_raw  - mu_t) / sd_t
#         s1 = (s1_raw - mu_t) / sd_t

#         r = (r / max(1e-6, reward_scale))
#         r = torch.clamp(r, -reward_clip, reward_clip)

#         # value target bootstrap with EMA target V_t
#         _, v1_t = target(s1)
#         td_target = r + gamma * (1.0 - done) * v1_t

#         # current
#         logits, v = model(s)
#         logits = logits.masked_fill(~mask, -1e9)
#         logp_all = F.log_softmax(logits, dim=-1)
#         logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)
#         pi = F.softmax(logits, dim=-1)

#         # behavior
#         b_logits, _ = behavior(s)
#         b_logits = b_logits.masked_fill(~mask, -1e9)
#         b_logp_all = F.log_softmax(b_logits, dim=-1)
#         b_logp = b_logp_all.gather(1, a.view(-1,1)).squeeze(1)

#         adv = (td_target - v).detach()
#         adv = adv / (adv.abs().mean().clamp_min(1e-6))

#         valid = (sw_eff > 0)

#         # clamp log-ratio to prevent exp overflow
#         log_ratio = (logp - b_logp).clamp(-20.0, 20.0)
#         valid = (sw_eff > 0)

#         log_ratio = (logp - b_logp).clamp(-20.0, 20.0)
#         ratio = torch.exp(log_ratio)

#         clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
#         obj1 = ratio * adv
#         obj2 = clipped * adv
#         surrogate = torch.minimum(obj1, obj2)

#         if valid.any():
#             den = sw_eff[valid].sum().clamp_min(1.0)
#             actor_loss = -(sw_eff[valid] * surrogate[valid]).sum() / den
#         else:
#             actor_loss = torch.zeros([], device=DEVICE)

#         vf_loss = F.smooth_l1_loss(v, td_target.detach())
#         ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()

#         approx_kl = (b_logp - logp).mean()  # rough KL(pi_b || pi)
#         clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

#         loss = actor_loss + vf_coef * vf_loss - ent_coef * ent

#         stats["loss"].append(float(loss.item()))
#         stats["actor"].append(float(actor_loss.item()))
#         stats["vf"].append(float(vf_loss.item()))
#         stats["ent"].append(float(ent.item()))
#         stats["approx_kl"].append(float(approx_kl.item()))
#         stats["clipfrac"].append(float(clipfrac.item()))

#     return {k: safe_mean(v) for k, v in stats.items()}


# # -----------------------------
# # Train
# # -----------------------------
# def train_loop(
#     train_csv: str,
#     val_csv: str,
#     epochs: int,
#     bc_epochs: int,
#     batch_size: int,
#     lr: float,
#     gamma: float,
#     seed: int,
#     clip_eps: float,
#     vf_coef: float,
#     ent_coef: float,
#     reward_clip: float,
#     reward_scale: float,
#     ema: float,
#     mix_hold: float, mix_cli: float, mix_cld: float,
#     boost_hold: float, boost_cli: float, boost_cld: float,
#     out_ckpt: str,
# ):
#     set_seed(seed)

#     ds_tr = TrajDataset(train_csv, scaler=None)
#     scaler = {"mu": ds_tr.mu, "sd": ds_tr.sd}
#     ds_va = TrajDataset(val_csv, scaler=scaler)

#     obs_dim = ds_tr.s.shape[1]
#     print(f"TRAIN: {len(ds_tr):,} rows | obs_dim={obs_dim} | device={DEVICE}")
#     print("TRAIN action counts:", {0:int(ds_tr.action_counts[0]),1:int(ds_tr.action_counts[1]),2:int(ds_tr.action_counts[2])})
#     print(f"VAL:   {len(ds_va):,} rows")
#     print("VAL action counts:", {0:int(ds_va.action_counts[0]),1:int(ds_va.action_counts[1]),2:int(ds_va.action_counts[2])})

#     sampler, mix, boost = build_weighted_sampler(ds_tr, mix_hold, mix_cli, mix_cld, boost_hold, boost_cli, boost_cld)
#     tr_loader = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0, pin_memory=(DEVICE=="cuda"))
#     va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
#     print("Sampler: mix=", tuple(mix.tolist()), "boost=", tuple(boost.tolist()))

#     mu_t = torch.as_tensor(ds_tr.mu, device=DEVICE)
#     sd_t = torch.as_tensor(ds_tr.sd, device=DEVICE)

#     # current model
#     model = PPOActorCritic(obs_dim).to(DEVICE)
#     # EMA target for V
#     target = PPOActorCritic(obs_dim).to(DEVICE)
#     target.load_state_dict(model.state_dict())
#     for p in target.parameters():
#         p.requires_grad_(False)

#     opt = torch.optim.Adam(model.parameters(), lr=lr)

#     # -------------------------
#     # Phase 1: behavior cloning warmup -> copy into behavior model
#     # -------------------------
#     print(f"\nPhase 1: Behavior Cloning warmup for {bc_epochs} epochs...")
#     for ep in range(1, bc_epochs + 1):
#         model.train()
#         ce_hist, ent_hist = [], []

#         for s_np, a_np, _, _, _, w_np in tr_loader:
#             s_raw = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32)
#             a     = torch.as_tensor(a_np, device=DEVICE, dtype=torch.long)
#             sw    = torch.as_tensor(w_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

#             s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=0.0, neginf=0.0)
#             mask  = compute_action_mask(s_raw, ds_tr.state_cols)
#             a_allowed = mask.gather(1, a.view(-1,1)).squeeze(1)
#             sw_eff = sw * a_allowed.float()

#             s = (s_raw - mu_t) / sd_t
#             logits, _ = model(s)
#             logits = logits.masked_fill(~mask, -1e9)

#             logp_all = F.log_softmax(logits, dim=-1)
#             logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)

#             den = sw_eff.sum().clamp_min(1.0)
#             ce = - (sw_eff * logp).sum() / den  # weighted NLL

#             pi = F.softmax(logits, dim=-1)
#             ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()

#             loss = ce - (ent_coef * ent)  # small entropy even in BC helps avoid collapse

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             ema_update(target, model, ema=ema)

#             ce_hist.append(float(ce.item()))
#             ent_hist.append(float(ent.item()))

#         print(f"BC Epoch {ep:02d} | nll={safe_mean(ce_hist):.4f} | ent={safe_mean(ent_hist):.4f}")

#     # frozen behavior policy = snapshot of model after BC
#     behavior = PPOActorCritic(obs_dim).to(DEVICE)
#     behavior.load_state_dict(model.state_dict())
#     for p in behavior.parameters():
#         p.requires_grad_(False)
#     behavior.eval()

#     # -------------------------
#     # Phase 2: Offline PPO updates
#     # -------------------------
#     print(f"\nPhase 2: Offline PPO for {epochs} epochs...")
#     best_val = float("inf")
#     os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)

#     for ep in range(1, epochs + 1):
#         model.train()
#         stats = {k: [] for k in ["loss","actor","vf","ent","approx_kl","clipfrac"]}
#         sampled_hist = np.zeros(3, dtype=np.int64)

#         for s_np, a_np, r_np, s1_np, done_np, w_np in tr_loader:
#             s_raw  = torch.as_tensor(s_np,  device=DEVICE, dtype=torch.float32)
#             s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
#             a      = torch.as_tensor(a_np,  device=DEVICE, dtype=torch.long)
#             r      = torch.as_tensor(r_np,  device=DEVICE, dtype=torch.float32)
#             done   = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
#             sw     = torch.as_tensor(w_np,  device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

#             sampled_hist[0] += int((a == 0).sum().item())
#             sampled_hist[1] += int((a == 1).sum().item())
#             sampled_hist[2] += int((a == 2).sum().item())

#             s_raw  = torch.nan_to_num(s_raw,  nan=0.0, posinf=0.0, neginf=0.0)
#             s1_raw = torch.nan_to_num(s1_raw, nan=0.0, posinf=0.0, neginf=0.0)
#             r      = torch.nan_to_num(r,      nan=0.0, posinf=0.0, neginf=0.0)

#             mask = compute_action_mask(s_raw, ds_tr.state_cols)
#             a_allowed = mask.gather(1, a.view(-1,1)).squeeze(1)
#             sw_eff = sw * a_allowed.float()

#             s  = (s_raw  - mu_t) / sd_t
#             s1 = (s1_raw - mu_t) / sd_t

#             # scaled+clipped reward
#             r = (r / max(1e-6, reward_scale))
#             r = torch.clamp(r, -reward_clip, reward_clip)

#             # TD target with EMA target critic
#             with torch.no_grad():
#                 _, v1_t = target(s1)
#                 td_target = r + gamma * (1.0 - done) * v1_t

#             # current policy/value
#             logits, v = model(s)
#             logits = logits.masked_fill(~mask, -1e9)
#             logp_all = F.log_softmax(logits, dim=-1)
#             logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)
#             pi = F.softmax(logits, dim=-1)

#             # behavior logp
#             with torch.no_grad():
#                 b_logits, _ = behavior(s)
#                 b_logits = b_logits.masked_fill(~mask, -1e9)
#                 b_logp_all = F.log_softmax(b_logits, dim=-1)
#                 b_logp = b_logp_all.gather(1, a.view(-1,1)).squeeze(1)

#             # advantage
#             adv = (td_target - v).detach()
#             adv = adv / (adv.abs().mean().clamp_min(1e-6))

#             # PPO ratio and clipped objective
#             ratio = torch.exp(logp - b_logp)
#             clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
#             obj1 = ratio * adv
#             obj2 = clipped * adv
#             surrogate = torch.minimum(obj1, obj2)

#             den = sw_eff.sum().clamp_min(1.0)
#             actor_loss = -(sw_eff * surrogate).sum() / den

#             vf_loss = F.smooth_l1_loss(v, td_target.detach())

#             ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
#             approx_kl = (b_logp - logp).mean()
#             clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

#             loss = actor_loss + vf_coef * vf_loss - ent_coef * ent

#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             ema_update(target, model, ema=ema)

#             stats["loss"].append(float(loss.item()))
#             stats["actor"].append(float(actor_loss.item()))
#             stats["vf"].append(float(vf_loss.item()))
#             stats["ent"].append(float(ent.item()))
#             stats["approx_kl"].append(float(approx_kl.item()))
#             stats["clipfrac"].append(float(clipfrac.item()))

#         samp_frac = sampled_hist / max(1, sampled_hist.sum())

#         # Validation
#         val_stats = eval_epoch(
#             model=model, behavior=behavior, target=target, loader=va_loader,
#             mu_t=mu_t, sd_t=sd_t, state_cols=ds_tr.state_cols,
#             gamma=gamma, reward_clip=reward_clip, reward_scale=reward_scale,
#             vf_coef=vf_coef, ent_coef=ent_coef, clip_eps=clip_eps
#         )

#         tr_loss = safe_mean(stats["loss"])
#         va_loss = val_stats["loss"]

#         print(
#             f"Epoch {ep:02d} | "
#             f"train_loss={tr_loss:.4f} actor={safe_mean(stats['actor']):.4f} vf={safe_mean(stats['vf']):.4f} "
#             f"ent={safe_mean(stats['ent']):.4f} kl={safe_mean(stats['approx_kl']):.4f} clipfrac={safe_mean(stats['clipfrac']):.3f} | "
#             f"val_loss={va_loss:.4f} val_actor={val_stats['actor']:.4f} val_vf={val_stats['vf']:.4f} "
#             f"val_kl={val_stats['approx_kl']:.4f} val_clipfrac={val_stats['clipfrac']:.3f} | "
#             f"sampled=[{samp_frac[0]:.3f},{samp_frac[1]:.3f},{samp_frac[2]:.3f}]"
#         )

#         if va_loss < best_val:
#             best_val = va_loss
#             bundle = {
#                 "model": model.state_dict(),
#                 "behavior": behavior.state_dict(),
#                 "target": target.state_dict(),
#                 "obs_dim": obs_dim,
#                 "state_cols": ds_tr.state_cols,
#                 "scaler_mean": ds_tr.mu,
#                 "scaler_std": ds_tr.sd,
#                 "train_action_counts": ds_tr.action_counts,
#                 "val_action_counts": ds_va.action_counts,
#                 "config": {
#                     "epochs": epochs, "bc_epochs": bc_epochs, "batch_size": batch_size, "lr": lr,
#                     "gamma": gamma, "clip_eps": clip_eps, "vf_coef": vf_coef, "ent_coef": ent_coef,
#                     "reward_clip": reward_clip, "reward_scale": reward_scale, "ema": ema,
#                     "mix": [float(mix_hold), float(mix_cli), float(mix_cld)],
#                     "boost": [float(boost_hold), float(boost_cli), float(boost_cld)],
#                     "best_val_loss": float(best_val),
#                 }
#             }
#             torch.save(bundle, out_ckpt)
#             print(f"✅ Saved BEST checkpoint (val_loss={best_val:.4f}) -> {out_ckpt}")

#     print("Done. Best val loss:", best_val)


# if __name__ == "__main__":
#     import argparse

#     p = argparse.ArgumentParser()
#     p.add_argument("--train-csv", required=True, help="Path to train dataset (.parquet/.pq or .csv, or a directory of shards)")
#     p.add_argument("--val-csv", required=True, help="Path to val dataset (.parquet/.pq or .csv, or a directory of shards)")
#     p.add_argument("--epochs", type=int, default=25)
#     p.add_argument("--bc-epochs", type=int, default=3)
#     p.add_argument("--batch-size", type=int, default=4096)
#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--gamma", type=float, default=0.99)
#     p.add_argument("--seed", type=int, default=42)

#     p.add_argument("--clip-eps", type=float, default=0.2)
#     p.add_argument("--vf-coef", type=float, default=0.5)
#     p.add_argument("--ent", type=float, default=0.002)

#     p.add_argument("--reward-clip", type=float, default=10.0)
#     p.add_argument("--reward-scale", type=float, default=100.0)
#     p.add_argument("--ema", type=float, default=0.995)

#     # Sampler
#     p.add_argument("--mix-hold", type=float, default=0.50)
#     p.add_argument("--mix-cli", type=float, default=0.25)
#     p.add_argument("--mix-cld", type=float, default=0.25)
#     p.add_argument("--boost-hold", type=float, default=1.0)
#     p.add_argument("--boost-cli", type=float, default=2.0)
#     p.add_argument("--boost-cld", type=float, default=3.0)

#     p.add_argument("--out", default="checkpoints/offline_ppo_3cls.pt")
#     args = p.parse_args()

#     train_loop(
#         train_csv=args.train_csv,
#         val_csv=args.val_csv,
#         epochs=args.epochs,
#         bc_epochs=args.bc_epochs,
#         batch_size=args.batch_size,
#         lr=args.lr,
#         gamma=args.gamma,
#         seed=args.seed,
#         clip_eps=args.clip_eps,
#         vf_coef=args.vf_coef,
#         ent_coef=args.ent,
#         reward_clip=args.reward_clip,
#         reward_scale=args.reward_scale,
#         ema=args.ema,
#         mix_hold=args.mix_hold, mix_cli=args.mix_cli, mix_cld=args.mix_cld,
#         boost_hold=args.boost_hold, boost_cli=args.boost_cli, boost_cld=args.boost_cld,
#         out_ckpt=args.out,
#     )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_offline_ppo_3cls.py
-------------------------
Offline PPO for 3 discrete actions (0=HOLD, 1=CLI, 2=CLD) using logged transitions.

Key idea (since dataset has no behavior log-probs):
1) Warmup: train a behavior cloning policy pi_b(a|s) on logged actions.
2) Freeze pi_b and run PPO updates on pi_theta using ratio exp(logp_new - logp_bc).
3) Critic is V(s) with EMA target V_t for TD bootstrap.
4) Action masking (guardrails) is applied to BOTH behavior and current policy logits.

Input Parquet/CSV:
- state features: columns starting with "s_"
- next state     : columns starting with "s1_"
- required cols  : action_id, reward, done
- optional       : sample_weight

Example:
python train_offline_ppo_3cls.py \
  --train-csv rl_dataset/merged/merged_train.csv \
  --val-csv   rl_dataset/merged/merged_val.csv \
  --epochs 25 --bc-epochs 3 --batch-size 4096 --lr 3e-4 \
  --gamma 0.99 --clip-eps 0.2 --vf-coef 0.5 --ent 0.002 \
  --mix-hold 0.50 --mix-cli 0.25 --mix-cld 0.25 \
  --out checkpoints/offline_ppo_3cls.pt
"""

import os, random, json
import numpy as np
import pandas as pd


from pathlib import Path
import glob

def read_table(path: str, low_memory: bool = False) -> pd.DataFrame:
    """Read a dataset from Parquet (preferred) or CSV.

    Supports:
      - single file: .parquet/.pq or .csv
      - directory: reads all *.parquet/*.pq (preferred) else *.csv inside the directory
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / "*.parquet")) + glob.glob(str(p / "*.pq")))
        if not files:
            files = sorted(glob.glob(str(p / "*.csv")))
        if not files:
            raise FileNotFoundError(f"No parquet/csv files found in directory: {path}")

        dfs = []
        for fp in files:
            ext = Path(fp).suffix.lower()
            if ext in [".parquet", ".pq"]:
                dfs.append(pd.read_parquet(fp))
            elif ext == ".csv":
                dfs.append(pd.read_csv(fp, low_memory=low_memory))
        if not dfs:
            raise FileNotFoundError(f"No readable parquet/csv files found in directory: {path}")
        return pd.concat(dfs, axis=0, ignore_index=True)

    ext = p.suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    if ext == ".csv":
        return pd.read_csv(p, low_memory=low_memory)

    raise ValueError(f"Unsupported input type: {path} (expected .parquet/.pq or .csv, or a directory)")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_HOLD, ACTION_CLI, ACTION_CLD = 0, 1, 2


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ema_update(target: nn.Module, source: nn.Module, ema: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(ema).add_(sp.data, alpha=(1.0 - ema))


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


# -----------------------------
# Metrics (no sklearn dependency)
# -----------------------------
def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < 3 and 0 <= p < 3:
            cm[int(t), int(p)] += 1
    return cm

def prf_from_cm(cm: np.ndarray):
    # returns per-class precision, recall, f1 plus macro/weighted and accuracy
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    true_pos = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall    = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)

    macro_f1 = float(np.mean(f1))
    weights = np.divide(true_pos, true_pos.sum(), out=np.zeros_like(true_pos), where=true_pos.sum() > 0)
    weighted_f1 = float(np.sum(weights * f1))
    acc = float(tp.sum() / max(1.0, cm.sum()))
    return precision, recall, f1, macro_f1, weighted_f1, acc


# -----------------------------
# Dataset
# -----------------------------
class TrajDataset(Dataset):
    def __init__(self, csv_path: str, state_prefix="s_", next_prefix="s1_", scaler=None):
        df = read_table(csv_path, low_memory=False).replace([np.inf, -np.inf], np.nan)

        self.state_cols = [c for c in df.columns if c.startswith(state_prefix)]
        self.next_cols  = [c for c in df.columns if c.startswith(next_prefix)]
        if not self.state_cols:
            raise RuntimeError("No state columns found (expected prefix 's_').")
        if not self.next_cols:
            raise RuntimeError("No next-state columns found (expected prefix 's1_').")

        required = ["reward", "done", "action_id"]
        for c in required:
            if c not in df.columns:
                raise RuntimeError(f"Missing required column: {c}")

        if "sample_weight" not in df.columns:
            df["sample_weight"] = 1.0

        # fill
        df[self.state_cols] = df[self.state_cols].fillna(0.0)
        df[self.next_cols]  = df[self.next_cols].fillna(0.0)
        for c in required + ["sample_weight"]:
            df[c] = df[c].fillna(0.0)

        self.s   = df[self.state_cols].astype(np.float32).to_numpy(copy=True)
        self.s1  = df[self.next_cols].astype(np.float32).to_numpy(copy=True)
        self.a   = df["action_id"].astype(np.int64).to_numpy(copy=True)
        self.r   = df["reward"].astype(np.float32).to_numpy(copy=True)
        self.d   = df["done"].astype(np.float32).clip(0.0, 1.0).to_numpy(copy=True)
        self.w   = df["sample_weight"].astype(np.float32).clip(0.0, 10.0).to_numpy(copy=True)

        self.action_counts = np.array([(self.a == 0).sum(), (self.a == 1).sum(), (self.a == 2).sum()], dtype=np.int64)

        # scaler (fit only on train, reuse on val)
        if scaler is None:
            mu = np.mean(self.s, axis=0).astype(np.float32)
            sd = np.std(self.s, axis=0).astype(np.float32)
            sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
            self.mu, self.sd = mu, sd
        else:
            self.mu = scaler["mu"].astype(np.float32)
            self.sd = scaler["sd"].astype(np.float32)

    def __len__(self):
        return int(self.a.shape[0])

    def __getitem__(self, idx: int):
        return self.s[idx], self.a[idx], self.r[idx], self.s1[idx], self.d[idx], self.w[idx]


# -----------------------------
# Model
# -----------------------------
class PPOActorCritic(nn.Module):
    """
    Actor: logits for Categorical over 3 actions.
    Critic: V(s) scalar.
    """
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, 3)
        self.v  = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.pi(x)             # [B,3]
        v = self.v(x).squeeze(-1)       # [B]
        return logits, v


# -----------------------------
# Guardrail Mask (RAW features)
# -----------------------------
def compute_action_mask(
    s_raw: torch.Tensor,
    state_cols: list,
    min_score_for_cli: float = 680.0,
    max_dpd_for_cli: float = 1.0,
    min_pay_ratio_for_cli: float = 0.90,
    min_util_for_cli: float = 0.25,
    max_util_for_cli: float = 0.95,
    min_overlimit_for_cld: float = 0.05,
    min_dpd_for_cld: float = 2.0,
    max_score_for_cld: float = 650.0
):
    """
    mask: [B,3] bool True=allowed.
    If required columns not present => allow all.
    """
    B = s_raw.shape[0]
    mask = torch.ones((B, 3), dtype=torch.bool, device=s_raw.device)

    def col(name):
        if name in state_cols:
            return s_raw[:, state_cols.index(name)]
        return None

    score = col("s_external_score")
    dpd   = col("s_dpd_count_12m")
    pay   = col("s_payment_ratio")
    utilm = col("s_max_utilization_6m")
    over  = col("s_overlimit_rate_90d")

    if score is None or dpd is None or pay is None or utilm is None:
        return mask

    cli_ok = (
        (score >= min_score_for_cli) &
        (dpd <= max_dpd_for_cli) &
        (pay >= min_pay_ratio_for_cli) &
        (utilm >= min_util_for_cli) &
        (utilm <= max_util_for_cli)
    )

    if over is None:
        cld_ok = (dpd >= min_dpd_for_cld) | (score <= max_score_for_cld)
    else:
        cld_ok = (dpd >= min_dpd_for_cld) | (over >= min_overlimit_for_cld) | (score <= max_score_for_cld)

    mask[:, ACTION_HOLD] = True
    mask[:, ACTION_CLI]  = cli_ok
    mask[:, ACTION_CLD]  = cld_ok
    return mask


# -----------------------------
# Sampler
# -----------------------------
def build_weighted_sampler(ds: TrajDataset, mix_hold, mix_cli, mix_cld, boost_hold, boost_cli, boost_cld):
    mix = np.array([mix_hold, mix_cli, mix_cld], dtype=np.float32)
    mix = mix / mix.sum()
    boost = np.array([boost_hold, boost_cli, boost_cld], dtype=np.float32)

    counts = ds.action_counts.astype(np.float64)
    counts = np.maximum(counts, 1.0)

    class_base = (mix.astype(np.float64) / counts) * boost.astype(np.float64)

    samp_w = np.zeros(len(ds), dtype=np.float64)
    samp_w[ds.a == ACTION_HOLD] = class_base[ACTION_HOLD]
    samp_w[ds.a == ACTION_CLI]  = class_base[ACTION_CLI]
    samp_w[ds.a == ACTION_CLD]  = class_base[ACTION_CLD]

    # include row weights
    samp_w = samp_w * ds.w.astype(np.float64)
    samp_w = np.clip(samp_w, 1e-12, None)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(samp_w),
        num_samples=len(ds),
        replacement=True,
    )
    return sampler, mix, boost


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def eval_epoch(model, behavior, target, loader, mu_t, sd_t, state_cols,
               gamma, reward_clip, reward_scale, vf_coef, ent_coef, clip_eps):
    model.eval()
    stats = {k: [] for k in ["loss","actor","vf","ent","approx_kl","clipfrac"]}
    y_true_all, y_pred_all = [], []
    y_true_valid, y_pred_valid = [], []

    for s_np, a_np, r_np, s1_np, done_np, w_np in loader:
        s_raw  = torch.as_tensor(s_np,  device=DEVICE, dtype=torch.float32)
        s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
        a      = torch.as_tensor(a_np,  device=DEVICE, dtype=torch.long)
        r      = torch.as_tensor(r_np,  device=DEVICE, dtype=torch.float32)
        done   = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
        sw     = torch.as_tensor(w_np,  device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

        s_raw  = torch.nan_to_num(s_raw,  nan=0.0, posinf=0.0, neginf=0.0)
        s1_raw = torch.nan_to_num(s1_raw, nan=0.0, posinf=0.0, neginf=0.0)
        r      = torch.nan_to_num(r,      nan=0.0, posinf=0.0, neginf=0.0)

        mask = compute_action_mask(s_raw, state_cols)
        a_allowed = mask.gather(1, a.view(-1, 1)).squeeze(1)
        sw_eff = sw * a_allowed.float()

        s  = (s_raw  - mu_t) / sd_t
        s1 = (s1_raw - mu_t) / sd_t

        r = (r / max(1e-6, reward_scale))
        r = torch.clamp(r, -reward_clip, reward_clip)

        # value target bootstrap with EMA target V_t
        _, v1_t = target(s1)
        td_target = r + gamma * (1.0 - done) * v1_t

        # current
        logits, v = model(s)
        logits = logits.masked_fill(~mask, -1e9)
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)
        pi = F.softmax(logits, dim=-1)

        # behavior
        b_logits, _ = behavior(s)
        b_logits = b_logits.masked_fill(~mask, -1e9)
        b_logp_all = F.log_softmax(b_logits, dim=-1)
        b_logp = b_logp_all.gather(1, a.view(-1,1)).squeeze(1)

        adv = (td_target - v).detach()
        adv = adv / (adv.abs().mean().clamp_min(1e-6))

        valid = (sw_eff > 0)

        # predictions for classification metrics (respecting mask)
        pred = torch.argmax(logits, dim=-1)
        y_true_all.append(a.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())
        if valid.any():
            y_true_valid.append(a[valid].detach().cpu().numpy())
            y_pred_valid.append(pred[valid].detach().cpu().numpy())

        # PPO ratio and clipped objective (safe under masking)
        log_ratio = (logp - b_logp).clamp(-20.0, 20.0)
        ratio = torch.exp(log_ratio)

        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        obj1 = ratio * adv
        obj2 = clipped * adv
        surrogate = torch.minimum(obj1, obj2)

        if valid.any():
            den = sw_eff[valid].sum().clamp_min(1.0)
            actor_loss = -(sw_eff[valid] * surrogate[valid]).sum() / den
        else:
            actor_loss = torch.zeros([], device=DEVICE)

        vf_loss = F.smooth_l1_loss(v, td_target.detach())
        ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()

        approx_kl = (b_logp - logp).mean()  # rough KL(pi_b || pi)
        clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

        loss = actor_loss + vf_coef * vf_loss - ent_coef * ent

        stats["loss"].append(float(loss.item()))
        stats["actor"].append(float(actor_loss.item()))
        stats["vf"].append(float(vf_loss.item()))
        stats["ent"].append(float(ent.item()))
        stats["approx_kl"].append(float(approx_kl.item()))
        stats["clipfrac"].append(float(clipfrac.item()))


    # F1 / accuracy metrics
    yt_all = np.concatenate(y_true_all) if len(y_true_all) else np.array([], dtype=np.int64)
    yp_all = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([], dtype=np.int64)
    cm_all = confusion_matrix_3(yt_all, yp_all) if yt_all.size else np.zeros((3,3), dtype=np.int64)
    p_all, r_all, f1_all, macro_f1_all, weighted_f1_all, acc_all = prf_from_cm(cm_all)
    
    yt_v = np.concatenate(y_true_valid) if len(y_true_valid) else np.array([], dtype=np.int64)
    yp_v = np.concatenate(y_pred_valid) if len(y_pred_valid) else np.array([], dtype=np.int64)
    cm_v = confusion_matrix_3(yt_v, yp_v) if yt_v.size else np.zeros((3,3), dtype=np.int64)
    p_v, r_v, f1_v, macro_f1_v, weighted_f1_v, acc_v = prf_from_cm(cm_v)
    
    out = {k: safe_mean(v) for k, v in stats.items()}
    out.update({
    "acc": acc_all,
    "macro_f1": macro_f1_all,
    "weighted_f1": weighted_f1_all,
    "acc_valid": acc_v,
    "macro_f1_valid": macro_f1_v,
    "weighted_f1_valid": weighted_f1_v,
    "cm": cm_all.tolist(),
    })
    return out


# -----------------------------
# Train
# -----------------------------
def train_loop(
    train_csv: str,
    val_csv: str,
    epochs: int,
    bc_epochs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    seed: int,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    reward_clip: float,
    reward_scale: float,
    ema: float,
    mix_hold: float, mix_cli: float, mix_cld: float,
    boost_hold: float, boost_cli: float, boost_cld: float,
    out_ckpt: str,
):
    set_seed(seed)

    ds_tr = TrajDataset(train_csv, scaler=None)
    scaler = {"mu": ds_tr.mu, "sd": ds_tr.sd}
    ds_va = TrajDataset(val_csv, scaler=scaler)

    obs_dim = ds_tr.s.shape[1]
    print(f"TRAIN: {len(ds_tr):,} rows | obs_dim={obs_dim} | device={DEVICE}")
    print("TRAIN action counts:", {0:int(ds_tr.action_counts[0]),1:int(ds_tr.action_counts[1]),2:int(ds_tr.action_counts[2])})
    print(f"VAL:   {len(ds_va):,} rows")
    print("VAL action counts:", {0:int(ds_va.action_counts[0]),1:int(ds_va.action_counts[1]),2:int(ds_va.action_counts[2])})

    sampler, mix, boost = build_weighted_sampler(ds_tr, mix_hold, mix_cli, mix_cld, boost_hold, boost_cli, boost_cld)
    tr_loader = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=0, pin_memory=(DEVICE=="cuda"))
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
    print("Sampler: mix=", tuple(mix.tolist()), "boost=", tuple(boost.tolist()))

    mu_t = torch.as_tensor(ds_tr.mu, device=DEVICE)
    sd_t = torch.as_tensor(ds_tr.sd, device=DEVICE)

    # current model
    model = PPOActorCritic(obs_dim).to(DEVICE)
    # EMA target for V
    target = PPOActorCritic(obs_dim).to(DEVICE)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Phase 1: behavior cloning warmup -> copy into behavior model
    # -------------------------
    print(f"\nPhase 1: Behavior Cloning warmup for {bc_epochs} epochs...")
    for ep in range(1, bc_epochs + 1):
        model.train()
        ce_hist, ent_hist = [], []

        for s_np, a_np, _, _, _, w_np in tr_loader:
            s_raw = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32)
            a     = torch.as_tensor(a_np, device=DEVICE, dtype=torch.long)
            sw    = torch.as_tensor(w_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

            s_raw = torch.nan_to_num(s_raw, nan=0.0, posinf=0.0, neginf=0.0)
            mask  = compute_action_mask(s_raw, ds_tr.state_cols)
            a_allowed = mask.gather(1, a.view(-1,1)).squeeze(1)
            sw_eff = sw * a_allowed.float()

            s = (s_raw - mu_t) / sd_t
            logits, _ = model(s)
            logits = logits.masked_fill(~mask, -1e9)

            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)

            den = sw_eff.sum().clamp_min(1.0)
            ce = - (sw_eff * logp).sum() / den  # weighted NLL

            pi = F.softmax(logits, dim=-1)
            ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()

            loss = ce - (ent_coef * ent)  # small entropy even in BC helps avoid collapse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(target, model, ema=ema)

            ce_hist.append(float(ce.item()))
            ent_hist.append(float(ent.item()))

        print(f"BC Epoch {ep:02d} | nll={safe_mean(ce_hist):.4f} | ent={safe_mean(ent_hist):.4f}")

    # frozen behavior policy = snapshot of model after BC
    behavior = PPOActorCritic(obs_dim).to(DEVICE)
    behavior.load_state_dict(model.state_dict())
    for p in behavior.parameters():
        p.requires_grad_(False)
    behavior.eval()

    # -------------------------
    # Phase 2: Offline PPO updates
    # -------------------------
    print(f"\nPhase 2: Offline PPO for {epochs} epochs...")
    best_macro_f1 = -1.0
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        stats = {k: [] for k in ["loss","actor","vf","ent","approx_kl","clipfrac"]}
        sampled_hist = np.zeros(3, dtype=np.int64)

        for s_np, a_np, r_np, s1_np, done_np, w_np in tr_loader:
            s_raw  = torch.as_tensor(s_np,  device=DEVICE, dtype=torch.float32)
            s1_raw = torch.as_tensor(s1_np, device=DEVICE, dtype=torch.float32)
            a      = torch.as_tensor(a_np,  device=DEVICE, dtype=torch.long)
            r      = torch.as_tensor(r_np,  device=DEVICE, dtype=torch.float32)
            done   = torch.as_tensor(done_np, device=DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
            sw     = torch.as_tensor(w_np,  device=DEVICE, dtype=torch.float32).clamp(0.0, 10.0)

            sampled_hist[0] += int((a == 0).sum().item())
            sampled_hist[1] += int((a == 1).sum().item())
            sampled_hist[2] += int((a == 2).sum().item())

            s_raw  = torch.nan_to_num(s_raw,  nan=0.0, posinf=0.0, neginf=0.0)
            s1_raw = torch.nan_to_num(s1_raw, nan=0.0, posinf=0.0, neginf=0.0)
            r      = torch.nan_to_num(r,      nan=0.0, posinf=0.0, neginf=0.0)

            mask = compute_action_mask(s_raw, ds_tr.state_cols)
            a_allowed = mask.gather(1, a.view(-1,1)).squeeze(1)
            sw_eff = sw * a_allowed.float()

            s  = (s_raw  - mu_t) / sd_t
            s1 = (s1_raw - mu_t) / sd_t

            # scaled+clipped reward
            r = (r / max(1e-6, reward_scale))
            r = torch.clamp(r, -reward_clip, reward_clip)

            # TD target with EMA target critic
            with torch.no_grad():
                _, v1_t = target(s1)
                td_target = r + gamma * (1.0 - done) * v1_t

            # current policy/value
            logits, v = model(s)
            logits = logits.masked_fill(~mask, -1e9)
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, a.view(-1,1)).squeeze(1)
            pi = F.softmax(logits, dim=-1)

            # behavior logp
            with torch.no_grad():
                b_logits, _ = behavior(s)
                b_logits = b_logits.masked_fill(~mask, -1e9)
                b_logp_all = F.log_softmax(b_logits, dim=-1)
                b_logp = b_logp_all.gather(1, a.view(-1,1)).squeeze(1)

            # advantage
            adv = (td_target - v).detach()
            adv = adv / (adv.abs().mean().clamp_min(1e-6))

            
            # PPO ratio and clipped objective (safe under masking)
            valid = (sw_eff > 0)

            log_ratio = (logp - b_logp).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)

            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            obj1 = ratio * adv
            obj2 = clipped * adv
            surrogate = torch.minimum(obj1, obj2)

            if valid.any():
                den = sw_eff[valid].sum().clamp_min(1.0)
                actor_loss = -(sw_eff[valid] * surrogate[valid]).sum() / den
            else:
                actor_loss = torch.zeros([], device=DEVICE)

            vf_loss = F.smooth_l1_loss(v, td_target.detach())

            ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
            approx_kl = (b_logp - logp).mean()
            clipfrac = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

            loss = actor_loss + vf_coef * vf_loss - ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(target, model, ema=ema)

            stats["loss"].append(float(loss.item()))
            stats["actor"].append(float(actor_loss.item()))
            stats["vf"].append(float(vf_loss.item()))
            stats["ent"].append(float(ent.item()))
            stats["approx_kl"].append(float(approx_kl.item()))
            stats["clipfrac"].append(float(clipfrac.item()))

        samp_frac = sampled_hist / max(1, sampled_hist.sum())

        # Validation
        val_stats = eval_epoch(
            model=model, behavior=behavior, target=target, loader=va_loader,
            mu_t=mu_t, sd_t=sd_t, state_cols=ds_tr.state_cols,
            gamma=gamma, reward_clip=reward_clip, reward_scale=reward_scale,
            vf_coef=vf_coef, ent_coef=ent_coef, clip_eps=clip_eps
        )

        tr_loss = safe_mean(stats["loss"])
        va_loss = val_stats["loss"]
        va_f1 = val_stats.get("macro_f1", 0.0)
        va_acc = val_stats.get("acc", 0.0)

        print(
            f"Epoch {ep:02d} | "
            f"train_loss={tr_loss:.4f} actor={safe_mean(stats['actor']):.4f} vf={safe_mean(stats['vf']):.4f} "
            f"ent={safe_mean(stats['ent']):.4f} kl={safe_mean(stats['approx_kl']):.4f} clipfrac={safe_mean(stats['clipfrac']):.3f} | "
            f"val_loss={va_loss:.4f} val_f1={va_f1:.4f} val_acc={va_acc:.4f} "
            f"val_actor={val_stats['actor']:.4f} val_vf={val_stats['vf']:.4f} "
            f"val_kl={val_stats['approx_kl']:.4f} val_clipfrac={val_stats['clipfrac']:.3f} | "
            f"sampled=[{samp_frac[0]:.3f},{samp_frac[1]:.3f},{samp_frac[2]:.3f}]"
        )

        if va_f1 > best_macro_f1:
            best_macro_f1 = va_f1
            bundle = {
                "model": model.state_dict(),
                "behavior": behavior.state_dict(),
                "target": target.state_dict(),
                "obs_dim": obs_dim,
                "state_cols": ds_tr.state_cols,
                "scaler_mean": ds_tr.mu,
                "scaler_std": ds_tr.sd,
                "train_action_counts": ds_tr.action_counts,
                "val_action_counts": ds_va.action_counts,
                "config": {
                    "epochs": epochs, "bc_epochs": bc_epochs, "batch_size": batch_size, "lr": lr,
                    "gamma": gamma, "clip_eps": clip_eps, "vf_coef": vf_coef, "ent_coef": ent_coef,
                    "reward_clip": reward_clip, "reward_scale": reward_scale, "ema": ema,
                    "mix": [float(mix_hold), float(mix_cli), float(mix_cld)],
                    "boost": [float(boost_hold), float(boost_cli), float(boost_cld)],
                    "best_macro_f1": float(best_macro_f1),
                    "best_val_loss_at_best_f1": float(va_loss),
                }
            }
            torch.save(bundle, out_ckpt)
            print(f"Saved BEST checkpoint (macro_f1={best_macro_f1:.4f}, val_loss={va_loss:.4f}) -> {out_ckpt}")

    print("Done. Best macro F1:", best_macro_f1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", required=True, help="Path to train dataset (.parquet/.pq or .csv, or a directory of shards)")
    p.add_argument("--val-parquet", required=True, help="Path to val dataset (.parquet/.pq or .csv, or a directory of shards)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bc-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.002)

    p.add_argument("--reward-clip", type=float, default=10.0)
    p.add_argument("--reward-scale", type=float, default=100.0)
    p.add_argument("--ema", type=float, default=0.995)

    # Sampler
    p.add_argument("--mix-hold", type=float, default=0.50)
    p.add_argument("--mix-cli", type=float, default=0.25)
    p.add_argument("--mix-cld", type=float, default=0.25)
    p.add_argument("--boost-hold", type=float, default=1.0)
    p.add_argument("--boost-cli", type=float, default=2.0)
    p.add_argument("--boost-cld", type=float, default=3.0)

    p.add_argument("--out_ckpt", default="checkpoints/offline_ppo_3cls.pt")
    args = p.parse_args()

    train_loop(
        train_csv=args.train_parquet,
        val_csv=args.val_parquet,
        epochs=args.epochs,
        bc_epochs=args.bc_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent,
        reward_clip=args.reward_clip,
        reward_scale=args.reward_scale,
        ema=args.ema,
        mix_hold=args.mix_hold, mix_cli=args.mix_cli, mix_cld=args.mix_cld,
        boost_hold=args.boost_hold, boost_cli=args.boost_cli, boost_cld=args.boost_cld,
        out_ckpt=args.out_ckpt,
    )
