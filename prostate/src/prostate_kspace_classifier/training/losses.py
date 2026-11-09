"""Loss helpers for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_with_logits(pos_weight: float | None = None) -> nn.Module:
    """Return a BCEWithLogitsLoss with optional positive class weighting."""

    weight = None
    if pos_weight is not None:
        weight = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing.
    
    Smooths targets: 0 -> smoothing/2, 1 -> 1 - smoothing/2
    This reduces overconfidence and improves calibration.
    """

    def __init__(self, smoothing: float = 0.1, pos_weight: float | None = None) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth labels: 0 -> eps, 1 -> 1-eps where eps = smoothing/2
        eps = self.smoothing / 2.0
        smoothed = targets * (1.0 - self.smoothing) + eps
        
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=pw)
        return F.binary_cross_entropy_with_logits(logits, smoothed)


def build_bce_loss(label_smoothing: float = 0.0, pos_weight: float | None = None) -> nn.Module:
    """Build BCE loss with optional label smoothing."""
    if label_smoothing > 0:
        return LabelSmoothingBCELoss(smoothing=label_smoothing, pos_weight=pos_weight)
    return bce_with_logits(pos_weight=pos_weight)


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """BCE with logits computed only on entries where ``mask`` is 1."""
    mask = mask.to(dtype=logits.dtype)
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    loss_per_item = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss_per_item * mask).sum() / denom
