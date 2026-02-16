from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .metrics import MODEL_LOAD_SECONDS
from types import SimpleNamespace

global _BUNDLE

@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    state_cols: List[str]
    scaler_mean: Optional[np.ndarray] = None
    scaler_std: Optional[np.ndarray] = None
    device: str = "cpu"


def _as_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x, dtype=np.float32)
    except Exception:
        return None

def _load_checkpoint(path: str):
    """
    PyTorch 2.6+ safe loader:
      - first tries weights_only=True (safe)
      - if checkpoint contains pickled objects/classes, retries weights_only=False
        (ONLY for trusted checkpoints, like your own training outputs)
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def _extract_state_cols(ckpt: Dict[str, Any]) -> List[str]:
    for key in ("state_cols", "feature_cols", "cols", "state_columns"):
        v = ckpt.get(key)
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v
    raise ValueError("Checkpoint missing state_cols/feature_cols. Store feature column order in the checkpoint.")


def _extract_scaler(ckpt: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mean = ckpt.get("scaler_mean", ckpt.get("mean"))
    std = ckpt.get("scaler_std", ckpt.get("std"))
    mean = _as_numpy(mean)
    std = _as_numpy(std)
    if mean is not None and std is not None and mean.shape == std.shape and mean.size > 0:
        return mean.astype(np.float32), std.astype(np.float32)
    return None, None


def _instantiate_model_from_ckpt(ckpt: Dict[str, Any]) -> torch.nn.Module:
    if isinstance(ckpt.get("model"), torch.nn.Module):
        return ckpt["model"]

    # If you saved a class path + init args, you can add that here later.
    # For now, require either model object or a callable builder in ckpt.
    builder = ckpt.get("model_builder")
    if callable(builder):
        return builder()

    raise ValueError("Checkpoint does not contain a torch.nn.Module under 'model' and no 'model_builder' callable found.")

def _instantiate_mag_beta_from_ckpt(ckpt: dict):
    import torch
    if not isinstance(ckpt, dict):
        raise ValueError("MAG checkpoint must be a dict.")

    sd = ckpt.get("model_state")
    if sd is None:
        raise ValueError("MAG checkpoint missing 'model_state'.")

    cfg = ckpt.get("config") or {}
    hidden = int(cfg.get("hidden", 256))
    depth  = int(cfg.get("depth", 3))

    feat_cols = ckpt.get("feature_cols")
    if not feat_cols:
        raise ValueError("MAG checkpoint missing 'feature_cols'.")

    obs_dim = len(feat_cols)

    # ✅ use the exact place where BetaRegressor is defined in your repo
    from src.modeling.magnitude.evaluate_mag import BetaRegressor

    model = BetaRegressor(obs_dim=obs_dim, hidden=hidden, depth=depth)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model



def load_model(path: str, kind: str, device: str = "cpu"):
    ckpt = _load_checkpoint(path)

    if kind in {"mag_cli", "mag_cld"}:
        model = _instantiate_mag_beta_from_ckpt(ckpt)  # RETURNS ONLY model
    else:
        model = _instantiate_model_from_ckpt(ckpt)

    model.to(device)
    model.eval()
    return model, ckpt


@dataclass
class ModelBundle:
    gate: LoadedModel
    dir: LoadedModel
    mag_cli: LoadedModel
    mag_cld: LoadedModel


_BUNDLE: Optional[ModelBundle] = None


def init_bundle(
    ckpt_root: str,
    device: str = "cpu",
    gate_file: str = "classification/gate_awac.pt",
    dir_file: str = "classification/dir_awac.pt",
    mag_cli_file: str = "regression/mag_cli_beta.pt",
    mag_cld_file: str = "regression/mag_cld_beta.pt",
) -> ModelBundle:
    global _BUNDLE
    start = time.perf_counter()

    #root = Path("/opt/airflow/app/checkpoints")
    root = Path(ckpt_root) 

    gate_model, gate_ckpt = load_model(str(root / gate_file), "gate", device=device)
    dir_model,  dir_ckpt  = load_model(str(root / dir_file),  "dir",  device=device)
    mag_cli_model, mag_cli_ckpt = load_model(str(root / mag_cli_file), "mag_cli", device=device)
    mag_cld_model, mag_cld_ckpt = load_model(str(root / mag_cld_file), "mag_cld", device=device)

    gate = SimpleNamespace(model=gate_model,device=device, **{k: v for k, v in gate_ckpt.items() if k != "model"})
    dir  = SimpleNamespace(model=dir_model,device=device,  **{k: v for k, v in dir_ckpt.items()  if k != "model"})

    mag_cli = SimpleNamespace(model=mag_cli_model,device=device, **mag_cli_ckpt)
    mag_cld = SimpleNamespace(model=mag_cld_model,device=device, **mag_cld_ckpt)

    # then return bundle with these objects
    bundle = SimpleNamespace(device=device, gate=gate, dir=dir, mag_cli=mag_cli, mag_cld=mag_cld)
    _BUNDLE = bundle
    return _BUNDLE




def get_bundle() -> ModelBundle:
    if _BUNDLE is None:
        raise RuntimeError("Model bundle not initialized. Call init_bundle() at startup.")
    return _BUNDLE
