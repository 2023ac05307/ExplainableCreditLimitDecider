from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GateModelConfig:
    """HOLD vs NONHOLD gate policy."""
    obs_dim: int
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.05
    layer_norm: bool = False


class MLPBackbone(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, depth: int, dropout: float, layer_norm: bool):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GateActorCritic(nn.Module):
    """
    Binary actor-critic:
      - policy head: logit for NONHOLD probability
      - optional q head (2 actions)
      - optional v head (scalar)

    We keep q/v because AWAC/IQL-style training often uses them,
    but inference only needs policy logit.
    """
    def __init__(self, cfg: GateModelConfig, *, include_q: bool = True, include_v: bool = True):
        super().__init__()
        self.cfg = cfg
        self.backbone = MLPBackbone(
            obs_dim=cfg.obs_dim,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
            layer_norm=cfg.layer_norm,
        )
        self.pi_logit = nn.Linear(cfg.hidden, 1)

        self.include_q = include_q
        self.include_v = include_v
        self.q = nn.Linear(cfg.hidden, 2) if include_q else None
        self.v = nn.Linear(cfg.hidden, 1) if include_v else None

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(obs)
        logit = self.pi_logit(h).squeeze(-1)  # [B]
        out: Dict[str, torch.Tensor] = {"logit": logit, "h": h}

        if self.q is not None:
            out["q"] = self.q(h)  # [B,2]
        if self.v is not None:
            out["v"] = self.v(h).squeeze(-1)  # [B]
        return out

    @torch.no_grad()
    def predict_proba(self, obs: torch.Tensor) -> torch.Tensor:
        """Return P(NONHOLD)."""
        self.eval()
        return torch.sigmoid(self.forward(obs)["logit"])


def safe_load_state_dict(model: nn.Module, state_dict: Dict[str, Any]) -> Tuple[bool, list[str], list[str]]:
    """
    Loads state_dict with strict=True if possible, else strict=False.
    Returns (strict_loaded, missing_keys, unexpected_keys).
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        return True, [], []
    except Exception:
        res = model.load_state_dict(state_dict, strict=False)
        # res is IncompatibleKeys(missing_keys, unexpected_keys)
        return False, list(res.missing_keys), list(res.unexpected_keys)
