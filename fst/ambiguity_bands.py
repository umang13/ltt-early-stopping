"""Time-normalized ambiguity bands and band-based stopping (same geometry as knee v2)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def find_optimal_delta(
    pos_scores: np.ndarray, neg_scores: np.ndarray, num_deltas: int
) -> Tuple[float, float, float]:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0, 1.0, 0.0
    deltas = np.linspace(0.0, 0.5, num_deltas)
    best_delta = 0.0
    best_b_pos = float(np.quantile(neg_scores, 1.0))
    best_b_neg = float(np.quantile(pos_scores, 0.0))
    for delta in deltas:
        b_pos = float(np.quantile(neg_scores, 1.0 - delta))
        b_neg = float(np.quantile(pos_scores, delta))
        if b_pos > b_neg:
            best_delta = float(delta)
            best_b_pos = b_pos
            best_b_neg = b_neg
    return best_delta, best_b_pos, best_b_neg


def compute_bands(
    probs: np.ndarray,
    label_matrix: np.ndarray,
    num_deltas: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    probs: (n, T, K) softmax probabilities.
    label_matrix: (n, K) binary (e.g. one-hot class indicators).
    """
    num_steps, num_labels = probs.shape[1], probs.shape[2]
    b_pos = np.zeros((num_steps, num_labels), dtype=float)
    b_neg = np.zeros((num_steps, num_labels), dtype=float)
    delta_star = np.zeros((num_steps, num_labels), dtype=float)
    for t in range(num_steps):
        for label_idx in range(num_labels):
            scores_t = probs[:, t, label_idx]
            labels_t = label_matrix[:, label_idx]
            pos_mask = labels_t == 1
            neg_mask = labels_t == 0
            neg_scores = scores_t[neg_mask]
            pos_scores_pos = scores_t[pos_mask]
            delta_star_t, b_pos_t, b_neg_t = find_optimal_delta(
                pos_scores_pos, neg_scores, num_deltas
            )
            delta_star[t, label_idx] = delta_star_t
            if np.any(neg_mask):
                b_pos[t, label_idx] = b_pos_t
            else:
                b_pos[t, label_idx] = 1.0
            if np.any(pos_mask):
                b_neg[t, label_idx] = b_neg_t
            else:
                b_neg[t, label_idx] = 0.0
    return b_pos, b_neg, delta_star


def compute_bounds(
    lam: float,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled ambiguity interval (lower, upper) per time and class."""
    lower = lam * b_neg + (1.0 - lam) * 0.0
    upper = lam * b_pos + (1.0 - lam) * 1.0
    return lower, upper


def compute_stop_indices(
    probs: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stop at the first t when every class probability lies outside (lower[t,k], upper[t,k]).

    probs: (n, T, K); lower, upper: (T, K) or broadcastable.
    If lower > upper for some (t,k), treat as immediately stoppable at that t.
    """
    num_samples, num_steps, _ = probs.shape
    stop_indices = np.full(num_samples, num_steps - 1, dtype=int)
    stopped = np.zeros(num_samples, dtype=bool)
    for t in range(num_steps):
        if np.any(lower[t] > upper[t]):
            stop_mask = np.ones(num_samples, dtype=bool)
        else:
            decide = (probs[:, t, :] < lower[t]) | (probs[:, t, :] > upper[t])
            stop_mask = decide.all(axis=1)
        newly_stopped = np.logical_and(~stopped, stop_mask)
        stop_indices[newly_stopped] = t
        stopped[newly_stopped] = True
    return stop_indices, stopped
