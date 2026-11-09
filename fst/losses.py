"""Sequence-level losses for calibration under dynamic stopping."""

from __future__ import annotations

import numpy as np


def per_sample_decisive_miscoverage(
    probs: np.ndarray,
    labels: np.ndarray,
    stop_indices: np.ndarray,
    stopped: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Miscoverage when the prediction set is the set of decisive classes:

        C = { k : p_k < lower[t,k] or p_k > upper[t,k] } at stop time t.

    Loss is 1{y not in C}, i.e. the true label's probability still lies in the
    ambiguity band [lower, upper] for that class index.

    If the trajectory never satisfies the stopping rule (stopped[i] is False),
    abstain with C = all classes and loss 0.
    """
    n = labels.shape[0]
    losses = np.zeros(n, dtype=float)
    if not np.any(stopped):
        return losses
    t = stop_indices[stopped]
    y = labels[stopped]
    p_true = probs[stopped, t, y]
    lo = lower[t, y]
    hi = upper[t, y]
    ambiguous = (p_true >= lo) & (p_true <= hi)
    losses[stopped] = ambiguous.astype(float)
    return losses
