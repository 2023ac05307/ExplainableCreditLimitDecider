from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DirModelConfig:
    """
    Direction policy on true NONHOLD:
      - output logit for CLI probability (1=CLI, 0=CLD)
    """
    obs_dim: int
    hidden: int = 256
    depth: int = 2
    dropout: float = 0.10   # ✅ midsem parity (was 0.05)
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


class DirActorCritic(nn.Module):
    """
    Binary actor-critic for direction:
      - policy head: logit for CLI probability (1=CLI, 0=CLD)
      - optional q head (2 actions)
      - optional v head (scalar)

    Inference only needs logit.
    """
    def __init__(self, cfg: DirModelConfig, *, include_q: bool = True, include_v: bool = True):
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
        logit = self.pi_logit(h).squeeze(-1)  # [B] => logit for P(CLI)
        out: Dict[str, torch.Tensor] = {"logit": logit, "h": h}
        if self.q is not None:
            out["q"] = self.q(h)  # [B,2]
        if self.v is not None:
            out["v"] = self.v(h).squeeze(-1)  # [B]
        return out

    @torch.no_grad()
    def predict_proba_cli(self, obs: torch.Tensor) -> torch.Tensor:
        """Return P(CLI)."""
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
        return False, list(res.missing_keys), list(res.unexpected_keys)
