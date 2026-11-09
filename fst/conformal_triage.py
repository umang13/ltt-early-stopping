"""
Scalar conformal-style triage thresholds (Angelopoulos et al., medRxiv 2024.02.09.24302543).

Port of ThresholdClassifier.calibrate (conformal-triage repo):

  • λ_hi: high-score side (fraction non-positive among scores > λ, binomial UC at ``tolerance``).
  • λ_lo: among scores ≤ λ, fraction diseased (y=1); one-sided UC on that selective error vs
    ``desired_npv`` (same invert/scan structure as reference).

Uses one probability per exam and task → ``b^- = λ_lo``, ``b^+ = λ_hi`` upstream of dilation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def estimate_scalar_triage_endpoints(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    desired_ppv: float,
    desired_npv: float,
    tolerance: float,
    num_thresh: int = 501,
    min_high_bin: int = 10,
    min_low_bin: int = 10,
) -> Tuple[float, float]:
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    lambdas = np.linspace(0.0, 1.0, max(int(num_thresh), 5))

    # ---------- λ_hi (PPV selective group: scores strictly > λ) ----------
    def selective_frac_nonich(lam: float) -> float:
        sel = s > lam
        nt = int(np.sum(sel))
        if nt == 0:
            return 0.0
        return float(nt - np.sum(y[sel])) / float(nt)

    def n_hi(lam: float) -> int:
        return int(np.sum(s > lam))

    lambdas_high = np.array(
        [lam for lam in lambdas if n_hi(lam) >= int(min_high_bin)], dtype=float
    )

    def selective_risk_ub_hi(lam_frak: float) -> float:
        n = int(n_hi(lam_frak))
        if n <= 0:
            return 0.0
        k = selective_frac_nonich(lam_frak) * float(n)

        def invert_for_ub(rr: float) -> float:
            return float(binom.cdf(k, n, rr) - float(tolerance))

        try:
            return float(brentq(invert_for_ub, 1e-10, 0.9999))
        except ValueError:
            return float(invert_for_ub(0.9999))

    lhat_hi = 1.0
    if lambdas_high.size > 0:
        step_unit = float(1.0 / max(float(lambdas_high.shape[0]), 1.0))
        for lhat_candidate in np.flip(lambdas_high):
            sr = selective_risk_ub_hi(lhat_candidate - step_unit)
            if sr > float(1.0 - desired_ppv):
                lhat_hi = float(lhat_candidate)
                break

    # ---------- λ_lo (NPV group: scores ≤ λ) ----------
    def selective_frac_disease_low(lam: float) -> float:
        sel = s <= lam
        nt = int(np.sum(sel))
        if nt == 0:
            return 0.0
        return float(np.sum(y[sel])) / float(nt)

    def n_lo(lam: float) -> int:
        return int(np.sum(s <= lam))

    lambdas_low = np.array(
        [lam for lam in lambdas if n_lo(lam) >= int(min_low_bin)], dtype=float
    )

    def selective_risk_ub_lo(lam_frak: float) -> float:
        n = int(n_lo(lam_frak))
        if n <= 0:
            return 0.0
        k = selective_frac_disease_low(lam_frak) * float(n)

        def invert_for_ub(rr: float) -> float:
            return float(binom.cdf(k, n, rr) - float(tolerance))

        try:
            return float(brentq(invert_for_ub, 1e-10, 0.9999))
        except ValueError:
            return float(invert_for_ub(0.9999))

    lhat_lo = 0.0
    if lambdas_low.size > 0:
        step_unit_lo = float(1.0 / max(float(lambdas_low.shape[0]), 1.0))
        for lhat_candidate in lambdas_low:
            sr = selective_risk_ub_lo(lhat_candidate + step_unit_lo)
            if sr > float(1.0 - desired_npv):
                lhat_lo = float(lhat_candidate)
                break

    return float(lhat_lo), float(lhat_hi)
