"""Fixed-sequence testing (FST) for selecting lambda on an ordered grid."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from fst.hoeffding import hoeffding_p_value_mean_gt_alpha


def fst_select_lambda(
    lambdas: np.ndarray,
    empirical_risks: np.ndarray,
    alpha: float,
    delta: float,
    B: float = 1.0,
    n: Optional[int] = None,
) -> Tuple[Optional[float], Dict]:
    """
    FST on hypotheses H_{0,j}: R(lambda_j) > alpha with p-values p_j.

    lambdas must be ordered from most conservative (j=0) to least (j=m-1).
    Reject H_{0,j} when p_j <= delta. Stop at the first k with p_k > delta and
    return lambda_{k-1}. If k=0, return None (no parameter passes the first test).

    empirical_risks[j] = mean_i ell_i(lambda_j) over the full calibration set,
    including abstentions with loss 0.

    n: calibration count; defaults to len(empirical_risks) is wrong — pass n_cal
    explicitly when risks are already means (scalar per j). Here empirical_risks
    should be shape (m,) of r_hat values and n is the number of samples used in each mean.
    """
    if lambdas.shape != empirical_risks.shape:
        raise ValueError("lambdas and empirical_risks must have the same shape")
    m = lambdas.shape[0]
    if n is None:
        raise ValueError("n (calibration size) is required")
    trace: List[Dict] = []
    for j in range(m):
        r_hat = float(empirical_risks[j])
        p_val = hoeffding_p_value_mean_gt_alpha(r_hat, int(n), float(alpha), B=float(B))
        trace.append({"j": j, "lambda": float(lambdas[j]), "r_hat": r_hat, "p_value": p_val})
        if p_val > float(delta):
            if j == 0:
                meta = {"trace": trace, "reason": "first_hypothesis_not_rejected"}
                return None, meta
            meta = {"trace": trace, "reason": "fst_stop", "k": j}
            return float(lambdas[j - 1]), meta
    meta = {"trace": trace, "reason": "all_rejected_return_most_aggressive"}
    return float(lambdas[m - 1]), meta


def empirical_risk_grid(
    loss_matrix: np.ndarray,
) -> np.ndarray:
    """loss_matrix (n, m): return (m,) column means."""
    return np.mean(loss_matrix, axis=0)


def fst_select_from_pvalues(
    lambdas: np.ndarray,
    p_values: np.ndarray,
    delta: float,
) -> Tuple[Optional[float], Dict]:
    """
    Fixed-sequence test driven by precomputed per-lambda p-values.

    lambdas must be ordered conservative (j=0) to aggressive (j=m-1). Reject
    H_{0,j} if p_j <= delta; stop at first non-rejection and return lambda_{k-1}.
    Returns None if k=0 (no parameter passes the first test) and the most
    aggressive lambda if all hypotheses are rejected.
    """
    if lambdas.shape != p_values.shape:
        raise ValueError("lambdas and p_values must have the same shape")
    m = lambdas.shape[0]
    trace: List[Dict] = []
    for j in range(m):
        p = float(p_values[j])
        trace.append({"j": j, "lambda": float(lambdas[j]), "p_value": p})
        if p > float(delta):
            if j == 0:
                return None, {"trace": trace, "reason": "first_hypothesis_not_rejected"}
            return float(lambdas[j - 1]), {"trace": trace, "reason": "fst_stop", "k": j}
    return float(lambdas[m - 1]), {"trace": trace, "reason": "all_rejected_return_most_aggressive"}
