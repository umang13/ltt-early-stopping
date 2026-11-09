"""Shared helpers for building (n, m) calibration loss matrices over lambda grids."""

from __future__ import annotations

import numpy as np

from fst.ambiguity_bands import compute_bounds, compute_stop_indices
from fst.losses import per_sample_decisive_miscoverage


def loss_matrix_band_decisive_miscoverage(
    probs: np.ndarray,
    labels: np.ndarray,
    lambdas: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
) -> np.ndarray:
    """
    probs: (n, T, K); labels: (n,) int; lambdas: (m,);
    b_pos, b_neg: (T, K) from estimation split.

    Returns loss_matrix (n, m) with abstain -> 0 loss.
    """
    n = labels.shape[0]
    m = lambdas.shape[0]
    out = np.zeros((n, m), dtype=float)
    for j in range(m):
        lower, upper = compute_bounds(float(lambdas[j]), b_pos, b_neg)
        stop_indices, stopped = compute_stop_indices(probs, lower, upper)
        out[:, j] = per_sample_decisive_miscoverage(
            probs, labels, stop_indices, stopped, lower, upper
        )
    return out
