from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import WeightedRandomSampler
except Exception:  # pragma: no cover
    torch = None
    WeightedRandomSampler = None


def build_weighted_sampler(
    dataset,
    *,
    mix0: float = 0.5,
    mix1: float = 0.5,
    boost0: float = 1.0,
    boost1: float = 1.0,
    replacement: bool = True,
    num_samples: Optional[int] = None,
) -> Optional["WeightedRandomSampler"]:
    """
    Trainer-compatible sampler builder.

    Weights per sample are class-based:
        w(class=0) ∝ (mix0 / count0) * boost0
        w(class=1) ∝ (mix1 / count1) * boost1

    Assumes dataset exposes dataset.a (numpy array of class labels).
    """
    if torch is None or WeightedRandomSampler is None:
        raise RuntimeError("PyTorch is required for build_weighted_sampler().")

    y = np.asarray(dataset.a).astype(np.int64).reshape(-1)
    n = int(len(y))
    if n == 0:
        return None

    c0 = int((y == 0).sum())
    c1 = int((y == 1).sum())

    # Avoid division by zero
    c0 = max(c0, 1)
    c1 = max(c1, 1)

    w0 = (float(mix0) / c0) * float(boost0)
    w1 = (float(mix1) / c1) * float(boost1)

    weights = np.where(y == 0, w0, w1).astype(np.float64)

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    if num_samples is None:
        num_samples = n

    return WeightedRandomSampler(weights=weights_t, num_samples=int(num_samples), replacement=bool(replacement))
