from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, Tuple, List

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = object


def _to_numpy(x) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _model_forward_to_prob(model, s: "torch.Tensor") -> "torch.Tensor":
    """
    Robustly extract probability of class=1 (NONHOLD) from model output.

    Supports model outputs:
      - Tensor (logits or probs)
      - tuple/list (first element)
      - dict with keys:
            'logit', 'logits',        # most common for BCE
            'pi',                     # policy head
            'prob', 'probs'            # already-sigmoid outputs
    """
    out = model(s)

    # ---- dict output (AWAC / PPO / IQL style models) ----
    if isinstance(out, dict):
        for key in ("logit", "logits", "pi", "prob", "probs"):
            if key in out:
                out = out[key]
                break
        else:
            raise KeyError(
                f"Model output dict keys {list(out.keys())} "
                "do not contain any of "
                "['logit','logits','pi','prob','probs']"
            )

    # ---- tuple / list ----
    if isinstance(out, (tuple, list)):
        out = out[0]

    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Model output must be torch.Tensor, got {type(out)}")

    # ---- logits vs probabilities ----
    if out.min().item() < 0.0 or out.max().item() > 1.0:
        return torch.sigmoid(out)

    return out



def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _metrics_from_conf(conf: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
    tpr = rec
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (tpr + tnr)
    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
    }


@dataclass
class GateEvalConfig:
    thresholds: Optional[List[float]] = None  # e.g. [0.05, 0.95, 19] or explicit list
    default_thr: float = 0.5


def _make_thresholds(thresholds: Optional[List[float]]) -> List[float]:
    """
    Supports:
      - None -> [0.5]
      - [start, end, n] -> linspace
      - explicit list -> as-is
    """
    if thresholds is None:
        return [0.5]
    if len(thresholds) == 3 and all(isinstance(x, (int, float)) for x in thresholds):
        a, b, n = float(thresholds[0]), float(thresholds[1]), int(thresholds[2])
        if n <= 1:
            return [a]
        return list(np.linspace(a, b, n))
    return [float(x) for x in thresholds]


@torch.no_grad()  # type: ignore[misc]
def eval_val_gate_style(
    model,
    val_loader: "DataLoader",
    mu_t=None,
    sd_t=None,
    thresholds=None,
    *,
    device: str = "cpu",
    return_best: bool = True,
) -> Dict[str, Any]:
    """
    Backward-compatible Gate evaluator.

    Supports both call styles:
      A) eval_val_gate_style(model, loader, mu_t, sd_t, thresholds)
      B) eval_val_gate_style(model, loader, device="cuda", thresholds=[...])

    Returns a FLAT dict compatible with train_gate_awac.py:
      {
        'thr', 'acc', 'prec', 'rec', 'f1', 'bal_acc',
        'tp', 'fp', 'fn', 'tn'
      }
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for eval_val_gate_style().")

    model.eval()

    # infer device
    if mu_t is not None and isinstance(mu_t, torch.Tensor):
        dev = mu_t.device
    else:
        dev = torch.device(device)

    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for batch in val_loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("val_loader must yield (s, y/a, ...) batches.")

        s = batch[0]
        y = batch[1]  # gate label: 0=HOLD, 1=NONHOLD

        s = s.to(dev)
        y = y.to(dev)

        # optional normalization
        if mu_t is not None and sd_t is not None:
            if isinstance(mu_t, torch.Tensor) and isinstance(sd_t, torch.Tensor):
                s = (s - mu_t) / sd_t

        p = _model_forward_to_prob(model, s)  # prob of NONHOLD
        probs_all.append(_to_numpy(p))
        y_all.append(_to_numpy(y))

    probs = np.concatenate(probs_all, axis=0).reshape(-1)
    y_true = np.concatenate(y_all, axis=0).reshape(-1).astype(np.int64)

    thr_list = _make_thresholds(thresholds)

    # baseline @0.5
    y_pred_05 = (probs >= 0.5).astype(np.int64)
    conf_05 = _confusion(y_true, y_pred_05)
    met_05 = _metrics_from_conf(conf_05)

    if not return_best:
        return {
            "thr": 0.5,
            **met_05,
            **conf_05,
        }

    best_thr = 0.5
    best_met = met_05
    best_conf = conf_05
    best_f1 = met_05["f1"]

    for thr in thr_list:
        y_pred = (probs >= float(thr)).astype(np.int64)
        conf = _confusion(y_true, y_pred)
        met = _metrics_from_conf(conf)
        if met["f1"] > best_f1:
            best_thr = float(thr)
            best_f1 = float(met["f1"])
            best_met = met
            best_conf = conf

    # ---- TRAINER-COMPATIBLE FLAT RETURN ----
    return {
        "thr": float(best_thr),
        "acc": float(best_met["acc"]),
        "prec": float(best_met["prec"]),
        "rec": float(best_met["rec"]),
        "f1": float(best_met["f1"]),
        "bal_acc": float(best_met["bal_acc"]),
        "tp": int(best_conf["tp"]),
        "fp": int(best_conf["fp"]),
        "fn": int(best_conf["fn"]),
        "tn": int(best_conf["tn"]),
        # optional diagnostics (won't break trainer)
        "_n": int(len(y_true)),
        "_pos_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "_thr_0.5": 0.5,
        "_metrics_0.5": met_05,
        "_conf_0.5": conf_05,
    }
