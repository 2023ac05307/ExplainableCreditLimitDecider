from __future__ import annotations

from typing import Iterable, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@torch.no_grad()  # type: ignore[misc]
def ema_update(
    target: "nn.Module",
    source: "nn.Module",
    *,
    ema: float = 0.995,
    update_buffers: bool = True,
    param_names: Optional[set[str]] = None,
) -> None:
    """
    Exponential Moving Average (EMA) update:
        target <- ema * target + (1-ema) * source

    Args:
        target: EMA model (updated in-place)
        source: live model
        ema: decay (closer to 1 => slower updates)
        update_buffers: if True, also copies buffers (BatchNorm running stats etc.)
        param_names: if provided, only update parameters whose names are in this set
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for ema_update().")

    if not (0.0 < ema < 1.0):
        raise ValueError(f"ema must be in (0,1), got {ema}")

    # Update parameters
    src_state = dict(source.named_parameters())
    for name, tgt_param in target.named_parameters():
        if param_names is not None and name not in param_names:
            continue
        src_param = src_state.get(name, None)
        if src_param is None:
            continue
        # Ensure dtype/device match
        tgt_param.data.mul_(ema).add_(src_param.data, alpha=(1.0 - ema))

    # Update buffers (optional)
    if update_buffers:
        src_buf = dict(source.named_buffers())
        for name, tgt_buf in target.named_buffers():
            s = src_buf.get(name, None)
            if s is None:
                continue
            # Buffers usually should be copied directly
            tgt_buf.copy_(s)
