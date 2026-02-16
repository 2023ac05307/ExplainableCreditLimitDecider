from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set seeds across Python, NumPy, and (optionally) PyTorch.

    Args:
        seed: int seed
        deterministic: If True, tries to make torch ops deterministic.
                       NOTE: Determinism may reduce performance and is not
                       guaranteed for all CUDA kernels.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # deterministic behavior (may slow down)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # older torch versions
                pass
        else:
            # faster default
            torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Use in DataLoader(worker_init_fn=seed_worker) for reproducible workers.
    """
    if torch is None:
        return

    # Each worker gets a different seed derived from the initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None):
    """
    Convenience: get a torch.Generator for DataLoader shuffling, etc.
    """
    if torch is None:
        return None
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g
