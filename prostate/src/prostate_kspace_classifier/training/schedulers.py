"""Helper utilities for simple epoch-wise LR schedules."""

from __future__ import annotations

import math
from typing import Callable, Optional


def _cosine_with_warmup_schedule(total_epochs: int, warmup_epochs: int) -> Callable[[int], float]:
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    def _schedule(epoch_idx: int) -> float:
        if epoch_idx < 1:
            raise ValueError("Epoch indices must start at 1")
        if warmup_epochs > 0 and epoch_idx <= warmup_epochs:
            return epoch_idx / warmup_epochs
        cosine_step = epoch_idx - warmup_epochs
        progress = (cosine_step - 1) / max(cosine_epochs - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _schedule


def create_epoch_scheduler(
    name: str,
    *,
    total_epochs: int,
    warmup_epochs: int,
) -> Optional[Callable[[int], float]]:
    if name == "none":
        return None
    if name == "cosine":
        return _cosine_with_warmup_schedule(total_epochs, warmup_epochs)
    raise ValueError(f"Unsupported lr scheduler '{name}'")
