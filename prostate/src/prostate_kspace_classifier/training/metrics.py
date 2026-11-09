"""Common evaluation metrics."""

from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import roc_auc_score


def _as_metric_vector(values: torch.Tensor) -> torch.Tensor:
    return values.detach().to(dtype=torch.float32, device="cpu").view(-1)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
) -> Dict[str, float]:
    """Return AUROC and accuracy."""
    if logits.numel() == 0 or targets.numel() == 0:
        return {"auroc": float("nan"), "accuracy": float("nan")}

    scores = _as_metric_vector(logits)
    labels_tensor = _as_metric_vector(targets)
    probs_tensor = torch.sigmoid(scores) if from_logits else scores
    probs = probs_tensor.numpy()
    labels = labels_tensor.numpy()
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    accuracy = float(((probs >= 0.5) == labels).mean())
    return {"auroc": auroc, "accuracy": accuracy}
