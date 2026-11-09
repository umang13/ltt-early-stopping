"""Hoeffding p-values for testing H0: E[L] > alpha with L in [0, B]."""

from __future__ import annotations

import math


def hoeffding_p_value_tail_mean_leq_c(
    sample_mean: float,
    n: int,
    boundary_c: float,
    span_b_minus_a: float,
) -> float:
    """
    One-sided Hoeffding tail: conservative upper bound on
    P(X_bar <= sample_mean | true mean = boundary_c), for i.i.d. bounded draws
    supported on an interval of length span_b_minus_a.

    Used for the W-statistic conditional test: W_i in a length-1 interval,
    test H0: E[W] > 0 at boundary E[W] = 0, small (negative) sample mean
    gives a small p-value.
    """
    if n <= 0 or span_b_minus_a <= 0.0:
        return 1.0
    # At boundary E[W]=0; only negative sample means contradict H0: E[W]>0.
    gap = float(sample_mean) - float(boundary_c)
    if gap >= 0.0:
        return 1.0
    exponent = -2.0 * int(n) * (gap**2) / (float(span_b_minus_a) ** 2)
    return min(1.0, float(math.exp(exponent)))


def hoeffding_p_value_mean_gt_alpha(
    r_hat: float,
    n: int,
    alpha: float,
    B: float = 1.0,
) -> float:
    """
    Conservative p-value for H0: true mean risk R > alpha vs small empirical mean.

    Uses the boundary mean mu = alpha: P(R_hat <= r_hat | mu = alpha) <=
    exp(-2 n (alpha - r_hat)^2 / B^2) for r_hat < alpha (one-sided Hoeffding).

    If r_hat >= alpha the data do not contradict H0, so p = 1.
    """
    if n <= 0 or B <= 0.0:
        return 1.0
    if r_hat >= float(alpha):
        return 1.0
    gap = float(alpha) - float(r_hat)
    exponent = -2.0 * int(n) * (gap**2) / (float(B) ** 2)
    return min(1.0, float(math.exp(exponent)))
