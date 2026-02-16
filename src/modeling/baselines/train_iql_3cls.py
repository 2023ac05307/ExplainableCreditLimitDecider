"""
train_iql_single_3cls_report.py  (PARQUET-enabled)
--------------------------------------------------
Same as your IQL 3-class script, but:
- Accepts Parquet file OR parquet dataset directory for train/val
- Keeps training logic + checkpoint bundle the same
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score


# ----------------------------
# Utils: scaler
# ----------------------------
def compute_scaler(X: np.ndarray):
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mu.astype(np.float32), std.astype(np.float32)

def apply_scaler(X: np.ndarray, mu: np.ndarray, std: np.ndarray):
    return ((X - mu) / std).astype(np.float32)

def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.is_dir():
        return pd.read_parquet(p)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    raise RuntimeError(f"Unsupported input format: {path} (use .csv, .parquet, or parquet directory)")

def to_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32, copy=True)


def _fmt_pred(pred_probs):
    return "[" + " ".join([f"{p:.3f}" for p in pred_probs]) + "]"

def print_val_block(tag, y_true, y_pred, labels=("HOLD","CLI","CLD"), print_cm=False):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    # metrics from cm
    P, R, F1, TP, FP, FN, TN = per_class_metrics_from_cm(cm)

    sup = cm.sum(axis=1)
    acc = np.trace(cm) / max(1, cm.sum())
    bal_acc = np.mean([R[i] for i in range(len(labels))])
    macro_f1 = np.mean([F1[i] for i in range(len(labels))])

    # pred distribution
    pred_dist = cm.sum(axis=0) / max(1, cm.sum())

    print(f"{tag} | acc={acc:.4f} macroF1={macro_f1:.4f} bal_acc={bal_acc:.4f} | pred={_fmt_pred(pred_dist)}")

    for i, name in enumerate(labels):
        print(f"  {name:<4} | P={P[i]:.3f} R={R[i]:.3f} F1={F1[i]:.3f} "
              f"(tp={TP[i]} fp={FP[i]} fn={FN[i]} tn={TN[i]})")

    if print_cm:
        print("  cm=")
        print(cm)

    return macro_f1, acc, bal_acc, pred_dist


# ----------------------------
# Metrics: multiclass "binary-style" per class
# ----------------------------
def multiclass_report(y_true, y_pred, n_classes=3):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    per_class = []
    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1v  = 2 * prec * rec / (prec + rec + 1e-9)

        per_class.append(dict(
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
            precision=float(prec), recall=float(rec), f1=float(f1v),
            support=int(cm[c, :].sum()),
        ))
    return cm, per_class



def per_class_metrics_from_cm(cm: np.ndarray):
    """Compute one-vs-rest P/R/F1 and TP/FP/FN/TN for each class from a multiclass confusion matrix."""
    cm = np.asarray(cm)
    n = cm.shape[0]
    total = cm.sum()
    P = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)
    F1 = np.zeros(n, dtype=np.float64)
    TP = np.zeros(n, dtype=np.int64)
    FP = np.zeros(n, dtype=np.int64)
    FN = np.zeros(n, dtype=np.int64)
    TN = np.zeros(n, dtype=np.int64)

    for c in range(n):
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(total - tp - fn - fp)

        TP[c], FN[c], FP[c], TN[c] = tp, fn, fp, tn
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1v  = 2.0 * prec * rec / (prec + rec + 1e-9)

        P[c], R[c], F1[c] = prec, rec, f1v

    return P, R, F1, TP, FP, FN, TN

def advantage_diagnostics(Qnet, Vnet, Xv, y_true, y_pred, device="cuda"):
    Qnet.eval(); Vnet.eval()
    with torch.no_grad():
        sv = torch.tensor(Xv, device=device)
        q_all = Qnet(sv)
        v = Vnet(sv).squeeze(-1)
        qmax = q_all.max(dim=1).values
        qt = q_all.gather(1, torch.tensor(y_true, device=device).view(-1,1)).squeeze(1)

        adv_all = (q_all - v.unsqueeze(1)).cpu().numpy()
        v_np = v.cpu().numpy()
        qmax_np = qmax.cpu().numpy()
        qt_np = qt.cpu().numpy()

    def grp_stats(mask, a_idx):
        if mask.sum() == 0:
            return (0, 0.0, 0.0)
        vals = adv_all[mask, a_idx]
        return (int(mask.sum()), float(vals.mean()), float(vals.std()))

    print(f"  [ADV] mean(V)={v_np.mean():.4f} | mean(Qmax)={qmax_np.mean():.4f} | mean(Q_true)={qt_np.mean():.4f}")

    for k, name in enumerate(["HOLD","CLI","CLD"]):
        m = (y_true == k)
        n, mu, sd = grp_stats(m, k)
        print(f"  [ADV][TRUE ] {name:4s} n={n:7d}  meanA={mu:+.4f}  stdA={sd:.4f}")

    for k, name in enumerate(["HOLD","CLI","CLD"]):
        m = (y_pred == k)
        n, mu, sd = grp_stats(m, k)
        print(f"  [ADV][PRED ] {name:4s} n={n:7d}  meanA={mu:+.4f}  stdA={sd:.4f}")


# ----------------------------
# Networks
# ----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(512,256), dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# IQL losses
# ----------------------------
def expectile_loss(diff: torch.Tensor, expectile: float):
    w = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (w * diff.pow(2)).mean()

def soft_update(target: nn.Module, online: nn.Module, tau: float):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))


def make_batches(n, batch_size):
    idx = np.random.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i:i+batch_size]


def main():
    ap = argparse.ArgumentParser()

    # Backward compatible flags (CSV)
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv", default=None)

    # New flags (Parquet)
    ap.add_argument("--train-parquet", default=None, help="Train parquet file OR dataset directory")
    ap.add_argument("--val-parquet", default=None, help="Val parquet file OR dataset directory")

    ap.add_argument("--state-prefix", default="s_")
    ap.add_argument("--next-prefix", default="s1_")
    ap.add_argument("--action-col", default="action_id")
    ap.add_argument("--reward-col", default="reward")
    ap.add_argument("--done-col", default="done")

    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--gamma", type=float, default=0.99)

    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--aw-temp", type=float, default=1.0)
    ap.add_argument("--aw-clip", type=float, default=20.0)

    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--q-target-tau", type=float, default=0.995)

    ap.add_argument("--reward-scale", type=float, default=1.0)
    ap.add_argument("--reward-clip", type=float, default=0.0)

    ap.add_argument("--hidden", type=str, default="512,256")
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--out", default="checkpoints/iql_single_3cls.pt")
    args = ap.parse_args()

    train_path = args.train_parquet or args.train_csv
    val_path = args.val_parquet or args.val_csv
    if not train_path or not val_path:
        raise RuntimeError("Provide train/val via --train-parquet/--val-parquet (preferred) or --train-csv/--val-csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    train_df = load_table(train_path)
    val_df   = load_table(val_path)

    state_cols = [c for c in train_df.columns if c.startswith(args.state_prefix)]
    next_cols  = [c for c in train_df.columns if c.startswith(args.next_prefix)]

    if len(state_cols) == 0:
        raise ValueError(f"No state columns found with prefix '{args.state_prefix}'")
    if len(next_cols) == 0:
        raise ValueError(f"No next-state columns found with prefix '{args.next_prefix}'")

    obs_dim = len(state_cols)

    print(f"TRAIN: {len(train_df):,} rows | obs_dim={obs_dim} | device={device}")
    print(f"State cols: {len(state_cols)} | Next cols: {len(next_cols)}")
    print("TRAIN counts:", train_df[args.action_col].value_counts().to_dict())
    print(f"VAL:   {len(val_df):,} rows")
    print("VAL counts:", val_df[args.action_col].value_counts().to_dict())
    print("Label map: 0=HOLD, 1=CLI, 2=CLD")

    # Extract arrays (numeric-safe)
    Xs  = to_numeric_matrix(train_df, state_cols)
    Xsp = to_numeric_matrix(train_df, next_cols)
    a   = pd.to_numeric(train_df[args.action_col], errors="coerce").fillna(0).to_numpy(np.int64)

    # reward / done
    if args.reward_col in train_df.columns:
        r = pd.to_numeric(train_df[args.reward_col], errors="coerce").fillna(0.0).to_numpy(np.float32) * float(args.reward_scale)
    else:
        r = np.zeros(len(train_df), dtype=np.float32)

    if args.reward_clip and args.reward_clip > 0:
        r = np.clip(r, -args.reward_clip, args.reward_clip)

    if args.done_col in train_df.columns:
        done = pd.to_numeric(train_df[args.done_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        done = np.zeros(len(train_df), dtype=np.float32)

    # Fit scaler on TRAIN states (apply to both s and s1)
    mu, std = compute_scaler(Xs)
    Xs  = apply_scaler(Xs,  mu, std)
    Xsp = apply_scaler(Xsp, mu, std)

    # VAL
    Xv  = apply_scaler(to_numeric_matrix(val_df, state_cols), mu, std)
    yv  = pd.to_numeric(val_df[args.action_col], errors="coerce").fillna(0).to_numpy(np.int64)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    Q = MLP(obs_dim, 3, hidden=hidden, dropout=args.dropout).to(device)
    Q_targ = MLP(obs_dim, 3, hidden=hidden, dropout=args.dropout).to(device)
    V = MLP(obs_dim, 1, hidden=hidden, dropout=args.dropout).to(device)
    Pi = MLP(obs_dim, 3, hidden=hidden, dropout=args.dropout).to(device)

    Q_targ.load_state_dict(Q.state_dict())

    optQ = optim.Adam(Q.parameters(), lr=args.lr)
    optV = optim.Adam(V.parameters(), lr=args.lr)
    optPi = optim.Adam(Pi.parameters(), lr=args.lr)

    ce = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1

    # ----------------------------
    # Warmup: Behavior cloning
    # ----------------------------
    if args.warmup_epochs > 0:
        print(f"\nPhase 1: Behavior Cloning warmup for {args.warmup_epochs} epochs...")
        for ep in range(1, args.warmup_epochs + 1):
            Pi.train()
            total_nll = 0.0
            total_ent = 0.0
            n_steps = 0

            for b in make_batches(len(Xs), args.batch_size):
                s = torch.tensor(Xs[b], device=device)
                ab = torch.tensor(a[b], device=device)

                logits = Pi(s)
                logp = torch.log_softmax(logits, dim=-1)
                nll = ce(logits, ab)
                ent = -(logp.exp() * logp).sum(-1).mean()

                loss = nll - args.ent_coef * ent
                optPi.zero_grad()
                loss.backward()
                optPi.step()

                total_nll += float(nll.item())
                total_ent += float(ent.item())
                n_steps += 1

            with torch.no_grad():
                sv = torch.tensor(Xv, device=device)
                pred = Pi(sv).argmax(dim=-1).cpu().numpy().astype(int)

            macro_f1, acc, bal_acc, _ = print_val_block(
                tag=f"[WARMUP] Ep {ep:02d} | train_nll={(total_nll/max(1,n_steps)):.4f} train_ent={(total_ent/max(1,n_steps)):.4f} | VAL",
                y_true=yv, y_pred=pred
            )
            print("")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_epoch = ep
                torch.save({
                    "algo": "IQL_3CLS",
                    "epoch": ep,
                    "scaler": {"mu": mu, "std": std, "state_cols": state_cols, "next_cols": next_cols},
                    "config": vars(args),
                    "Q": Q.state_dict(),
                    "V": V.state_dict(),
                    "Pi": Pi.state_dict(),
                    "best_macro_f1": best_macro_f1,
                }, args.out)
                print(f"Saved BEST checkpoint by macroF1={macro_f1:.4f} -> {args.out}")

    # ----------------------------
    # IQL Training
    # ----------------------------
    print(f"\nPhase 2: Offline IQL for {args.epochs} epochs...")
    for ep in range(1, args.epochs + 1):
        Q.train(); V.train(); Pi.train()

        q_loss_acc = 0.0
        v_loss_acc = 0.0
        pi_loss_acc = 0.0
        ent_acc = 0.0
        n_steps = 0
        adv_sum = 0.0
        w_sum = 0.0
        n_wsteps = 0

        for b in make_batches(len(Xs), args.batch_size):
            s  = torch.tensor(Xs[b],  device=device)
            sp = torch.tensor(Xsp[b], device=device)
            ab = torch.tensor(a[b],   device=device)
            rb = torch.tensor(r[b],   device=device)
            db = torch.tensor(done[b],device=device)

            with torch.no_grad():
                v_sp = V(sp).squeeze(-1)
                target = rb + args.gamma * (1.0 - db) * v_sp

            q_all = Q(s)
            q_sa = q_all.gather(1, ab[:,None]).squeeze(1)
            q_loss = (q_sa - target).pow(2).mean()

            optQ.zero_grad()
            q_loss.backward()
            optQ.step()

            with torch.no_grad():
                q_all_det = Q(s)
                q_sa_det = q_all_det.gather(1, ab[:,None]).squeeze(1)

            v_s = V(s).squeeze(-1)
            diff = q_sa_det - v_s
            v_loss = expectile_loss(diff, args.expectile)

            optV.zero_grad()
            v_loss.backward()
            optV.step()

            with torch.no_grad():
                v_s2 = V(s).squeeze(-1)
                q_all2 = Q(s)
                q_sa2 = q_all2.gather(1, ab[:,None]).squeeze(1)
                adv = (q_sa2 - v_s2)
                w = torch.exp(adv / max(1e-6, args.aw_temp)).clamp(max=args.aw_clip)
                adv_sum += float(adv.mean().item()); w_sum += float(w.mean().item()); n_wsteps += 1

            logits = Pi(s)
            logp = torch.log_softmax(logits, dim=-1)
            logp_a = logp.gather(1, ab[:,None]).squeeze(1)
            ent = -(logp.exp() * logp).sum(-1).mean()

            pi_loss = -(w.detach() * logp_a).mean() - args.ent_coef * ent

            optPi.zero_grad()
            pi_loss.backward()
            optPi.step()

            soft_update(Q_targ, Q, args.q_target_tau)

            q_loss_acc += float(q_loss.item())
            v_loss_acc += float(v_loss.item())
            pi_loss_acc += float(pi_loss.item())
            ent_acc += float(ent.item())
            n_steps += 1

        with torch.no_grad():
            sv = torch.tensor(Xv, device=device)
            pred = Pi(sv).argmax(dim=-1).cpu().numpy().astype(int)

        macro_f1, acc, bal_acc, cm = print_val_block(
            tag=(f"[IQL] Ep {ep:02d} | "
                 f"q_loss={(q_loss_acc/max(1,n_steps)):.4f} "
                 f"v_loss={(v_loss_acc/max(1,n_steps)):.4f} "
                 f"pi_loss={(pi_loss_acc/max(1,n_steps)):.4f} "
                 f"ent={(ent_acc/max(1,n_steps)):.4f} | VAL"),
            y_true=yv, y_pred=pred
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = ep
            torch.save({
                "algo": "IQL_3CLS",
                "epoch": ep,
                "scaler": {"mu": mu, "std": std, "state_cols": state_cols, "next_cols": next_cols},
                "config": vars(args),
                "Q": Q.state_dict(),
                "V": V.state_dict(),
                "Pi": Pi.state_dict(),
                "best_macro_f1": best_macro_f1,
            }, args.out)
            print(f"Saved BEST checkpoint by macroF1={macro_f1:.4f} -> {args.out}")

    print(f"\nDone. Best macroF1={best_macro_f1:.4f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
