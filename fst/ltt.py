"""Learn-Then-Test (LTT) with Holm step-down for FWER-controlled lambda selection."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from fst.hoeffding import (
    hoeffding_p_value_mean_gt_alpha,
    hoeffding_p_value_tail_mean_leq_c,
)

AlphaSpec = Union[float, Sequence[float], np.ndarray]


def _as_alpha_vector(alpha: AlphaSpec, n_labels: int) -> np.ndarray:
    """Broadcast scalar α or validate length-K α_k vector."""
    if np.isscalar(alpha):
        return np.full(int(n_labels), float(alpha), dtype=float)
    a = np.asarray(alpha, dtype=float).ravel()
    if a.size != int(n_labels):
        raise ValueError(
            f"alpha_per_label length {a.size} != number of labels {n_labels}"
        )
    return a


def binomial_pvalue_mean_gt_alpha(
    n_fail: int,
    n: int,
    alpha: float,
    *,
    method: str = "binomial",
) -> float:
    """
    Conservative p-value for H0: R > alpha with binary losses (exact Binomial CDF).

    With k = n_fail failures out of n trials and empirical R = k/n:
      p = P(Bin(n, alpha) <= k) = binom.cdf(k, n, alpha)
    Returns 1 if n == 0 or R >= alpha.

    ``method="hoeffding"`` uses the Hoeffding upper bound instead (fallback).
    """
    n_i = int(n)
    k = int(n_fail)
    a = float(alpha)
    if n_i <= 0:
        return 1.0
    if k < 0:
        k = 0
    if k > n_i:
        k = n_i
    r_hat = float(k) / float(n_i)
    if r_hat >= a:
        return 1.0
    if method == "hoeffding":
        return hoeffding_p_value_mean_gt_alpha(r_hat, n_i, a, B=1.0)
    if method != "binomial":
        raise ValueError(f"Unknown p-value method: {method}")
    if a <= 0.0:
        # Under H0: R > 0, observing k failures never rejects when alpha<=0.
        return 1.0
    if a >= 1.0:
        # H0: R > 1 is empty; observing any R_hat <= 1 never rejects → p = 1.
        return 1.0
    try:
        from scipy.stats import binom

        return float(min(1.0, binom.cdf(k, n_i, a)))
    except ImportError:
        # Stable-ish fallback via multiplicative PMF sum (OK for moderate n).
        # P(X<=k) = sum_{i=0}^k C(n,i) a^i (1-a)^{n-i}
        log_terms = []
        log_a = math.log(a)
        log_1ma = math.log1p(-a)
        for i in range(0, k + 1):
            log_terms.append(
                math.lgamma(n_i + 1)
                - math.lgamma(i + 1)
                - math.lgamma(n_i - i + 1)
                + i * log_a
                + (n_i - i) * log_1ma
            )
        m = max(log_terms)
        s = sum(math.exp(t - m) for t in log_terms)
        return float(min(1.0, math.exp(m) * s))


def class_conditional_label_risk(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped: np.ndarray,
    label_idx: int,
    *,
    risk_kind: str,
) -> Tuple[float, int, int]:
    """
    One label's class-conditional binary risk on stopped samples.

    FNR: S_pos = stopped & y_k==1; R = mean(pred_k==0), returns (R, n, n_fail).
    FPR: S_neg = stopped & y_k==0; R = mean(pred_k==1).
    If n==0 returns (0.0, 0, 0).
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    stopped = np.asarray(stopped, dtype=bool).ravel()
    k = int(label_idx)
    yk = labels[:, k].astype(np.int64, copy=False)
    pk = preds[:, k].astype(np.int64, copy=False)
    if risk_kind == "fnr":
        mask = stopped & (yk == 1)
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0, 0
        n_fail = int(np.sum(pk[mask] == 0))
    elif risk_kind == "fpr":
        mask = stopped & (yk == 0)
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0, 0
        n_fail = int(np.sum(pk[mask] == 1))
    else:
        raise ValueError(f"Unknown risk_kind: {risk_kind}")
    return float(n_fail) / float(n), n, n_fail


def class_conditional_risks_pvalues_one_lambda(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped: np.ndarray,
    alpha: AlphaSpec,
    *,
    risk_metric: str,
    pvalue_method: str = "binomial",
    control_both: bool = False,
    control_label_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """
    Option (i): per-label class-conditional risks and p-values for one λ.

    ``p(λ) = max_k p_k`` over controlled labels (and FPR components too if ``control_both``).
    Scalar reporting risk is ``max_k R_k`` over controlled components with n_k > 0 (else 0).

    ``control_label_indices``: subset of label indices included in the max; ``None`` = all K.
    Stopping / predictions may still use the full label vector upstream.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n_labels = int(labels.shape[1])
    alpha_k = _as_alpha_vector(alpha, n_labels)

    if control_label_indices is None:
        labs = list(range(n_labels))
    else:
        labs = [int(i) for i in control_label_indices]
        if not labs:
            raise ValueError("control_label_indices is empty")
        for i in labs:
            if i < 0 or i >= n_labels:
                raise ValueError(
                    f"control_label_indices entry {i} out of range for K={n_labels}"
                )

    kinds: List[str]
    if control_both:
        kinds = ["fnr", "fpr"]
    elif risk_metric in ("fnr", "fpr"):
        kinds = [risk_metric]
    else:
        raise ValueError(f"Unknown risk_metric: {risk_metric}")

    risks_k: List[float] = []
    ns_k: List[int] = []
    fails_k: List[int] = []
    p_k: List[float] = []
    kind_k: List[str] = []
    label_ids: List[int] = []

    for kind in kinds:
        a_vec = alpha_k
        for lab in labs:
            r, n, n_fail = class_conditional_label_risk(
                preds, labels, stopped, lab, risk_kind=kind
            )
            p = binomial_pvalue_mean_gt_alpha(
                n_fail, n, float(a_vec[lab]), method=pvalue_method
            )
            risks_k.append(r)
            ns_k.append(n)
            fails_k.append(n_fail)
            p_k.append(p)
            kind_k.append(kind)
            label_ids.append(lab)

    p_arr = np.asarray(p_k, dtype=float)
    r_arr = np.asarray(risks_k, dtype=float)
    n_arr = np.asarray(ns_k, dtype=int)
    positive_n = n_arr > 0
    p_combined = float(np.max(p_arr)) if p_arr.size else 1.0
    if np.any(positive_n):
        r_report = float(np.max(r_arr[positive_n]))
        n_report = int(np.min(n_arr[positive_n]))
    else:
        r_report = 0.0
        n_report = 0

    argmax_p = int(np.argmax(p_arr)) if p_arr.size else -1

    return {
        "p_value": p_combined,
        "r_hat": r_report,
        "n_used": n_report,
        "risks_per_component": r_arr.tolist(),
        "ns_per_component": n_arr.tolist(),
        "fails_per_component": [int(x) for x in fails_k],
        "p_per_component": p_arr.tolist(),
        "kinds": kind_k,
        "label_ids": label_ids,
        "alpha_per_label": alpha_k.tolist(),
        "argmax_p_component": argmax_p,
        "argmax_p_label_id": int(label_ids[argmax_p]) if argmax_p >= 0 else None,
        "control_label_indices": list(labs),
    }


def grid_class_conditional_pvalues(
    preds_cube: np.ndarray,
    labels: np.ndarray,
    stopped_mat: np.ndarray,
    alpha: AlphaSpec,
    *,
    risk_metric: str,
    pvalue_method: str = "binomial",
    control_both: bool = False,
    control_label_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Per-λ Option (i) p-values / reporting risks / n.

    ``preds_cube`` shape (n, m, K), ``stopped_mat`` (n, m), ``labels`` (n, K).

    Returns (p_values, risks, ns, per_lambda_details).
    """
    preds_cube = np.asarray(preds_cube)
    labels = np.asarray(labels)
    stopped_mat = np.asarray(stopped_mat, dtype=bool)
    n, m, k = preds_cube.shape
    if labels.shape != (n, k):
        raise ValueError(
            f"labels shape {labels.shape} incompatible with preds_cube {preds_cube.shape}"
        )
    if stopped_mat.shape != (n, m):
        raise ValueError(
            f"stopped_mat shape {stopped_mat.shape} incompatible with preds_cube {preds_cube.shape}"
        )

    p_values = np.ones(m, dtype=float)
    risks = np.zeros(m, dtype=float)
    ns = np.zeros(m, dtype=int)
    details: List[Dict[str, object]] = []
    for j in range(m):
        d = class_conditional_risks_pvalues_one_lambda(
            preds_cube[:, j, :],
            labels,
            stopped_mat[:, j],
            alpha,
            risk_metric=risk_metric,
            pvalue_method=pvalue_method,
            control_both=control_both,
            control_label_indices=control_label_indices,
        )
        p_values[j] = float(d["p_value"])
        risks[j] = float(d["r_hat"])
        ns[j] = int(d["n_used"])
        details.append(d)
    return p_values, risks, ns, details


def holm_step_down(p_values: np.ndarray, delta: float) -> np.ndarray:
    """
    Holm's step-down procedure for FWER control at level delta.

    Returns a boolean mask of shape (m,) with True at rejected indices.

    Algorithm:
      1. Sort p-values ascending: p_(1) <= ... <= p_(m).
      2. For j = 1, ..., m, reject H_(j) if p_(j) <= delta / (m - j + 1).
      3. Stop at the first non-rejection; reject all earlier H_(k), k < j.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    rejected = np.zeros(m, dtype=bool)
    if m == 0:
        return rejected
    order = np.argsort(p)
    for rank, idx in enumerate(order, start=1):
        threshold = delta / (m - rank + 1)
        if p[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def empirical_risk_unconditional(
    losses_stopped: np.ndarray,
    stopped_mask: np.ndarray,
    abstain_penalty: float,
) -> Tuple[float, int]:
    """
    Unconditional risk with abstention penalty.

    losses_stopped[i] is defined for stopped samples (zero elsewhere).
    Abstained samples (stopped_mask is False) contribute abstain_penalty.

    Returns (r_hat, n_used) with n_used = total calibration size.
    """
    n = int(stopped_mask.shape[0])
    if n == 0:
        return 0.0, 0
    contrib = np.where(stopped_mask, losses_stopped, float(abstain_penalty))
    return float(np.mean(contrib)), n


def empirical_risk_conditional_on_stopped(
    losses_stopped: np.ndarray,
    stopped_mask: np.ndarray,
) -> Tuple[float, int]:
    """
    Conditional risk averaged over stopped samples only.

    Returns (r_hat, n_stopped). If n_stopped == 0, returns (0.0, 0).
    """
    idx = np.where(stopped_mask)[0]
    if idx.size == 0:
        return 0.0, 0
    return float(np.mean(losses_stopped[idx])), int(idx.size)


def grid_risks_and_ns(
    losses_mat: np.ndarray,
    stopped_mat: np.ndarray,
    risk_mode: str,
    abstain_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-lambda r_hat (m,) and effective sample size n (m,).

    risk_mode:
      - "unconditional_with_abstain": R = mean over all calibration samples,
        with abstainers (stopped_mat False) contributing abstain_penalty;
        n = total calibration size.
      - "conditional_on_stopped": R = mean over stopped samples only;
        n = number of stopped samples (0 marks a degenerate test).
    """
    n_total, m = losses_mat.shape
    risks = np.zeros(m, dtype=float)
    ns = np.zeros(m, dtype=int)
    for j in range(m):
        l = losses_mat[:, j]
        s = stopped_mat[:, j]
        if risk_mode == "unconditional_with_abstain":
            contrib = np.where(s, l, float(abstain_penalty))
            risks[j] = float(np.mean(contrib))
            ns[j] = int(n_total)
        elif risk_mode == "conditional_on_stopped":
            idx = np.where(s)[0]
            if idx.size == 0:
                risks[j] = 0.0
                ns[j] = 0
            else:
                risks[j] = float(np.mean(l[idx]))
                ns[j] = int(idx.size)
        else:
            raise ValueError(f"Unknown risk_mode: {risk_mode}")
    return risks, ns


def _macro_balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-average over labels of (sensitivity + specificity) / 2 on a cohort."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if preds.size == 0 or labels.shape[0] == 0:
        return 0.0
    n_labels = int(labels.shape[1])
    bas: List[float] = []
    for k in range(n_labels):
        p = preds[:, k].astype(np.int64, copy=False)
        y = labels[:, k].astype(np.int64, copy=False)
        tp = int(np.sum((p == 1) & (y == 1)))
        tn = int(np.sum((p == 0) & (y == 0)))
        fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bas.append(0.5 * (sens + spec))
    return float(np.mean(bas)) if bas else 0.0


def grid_utilities(
    stopped_mat: np.ndarray,
    stop_idx_mat: np.ndarray,
    risks: np.ndarray,
    utility: str,
    *,
    preds_cube: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-lambda utility (higher is better) used to pick within the LTT safe set.
      - "stopped_count": number of stopped calibration samples.
      - "neg_mean_stop_index": negative mean stop index over all calibration
        samples (non-stopped contribute the horizon T-1 by construction).
      - "neg_calib_risk": -r_hat; useful when risks share slack and we want
        the safest passing lambda.
      - "balanced_accuracy_stopped": macro balanced accuracy of predictions on
        stopped calibration samples only (requires ``preds_cube``, ``labels``).
    """
    if utility == "stopped_count":
        return stopped_mat.sum(axis=0).astype(float)
    if utility == "neg_mean_stop_index":
        return -stop_idx_mat.mean(axis=0).astype(float)
    if utility == "neg_calib_risk":
        return -np.asarray(risks, dtype=float)
    if utility == "balanced_accuracy_stopped":
        if preds_cube is None or labels is None:
            raise ValueError(
                "utility='balanced_accuracy_stopped' requires preds_cube and labels"
            )
        preds_cube = np.asarray(preds_cube)
        labels = np.asarray(labels)
        n, m = stopped_mat.shape
        if preds_cube.shape[:2] != (n, m):
            raise ValueError(
                f"preds_cube shape {preds_cube.shape} incompatible with stopped_mat {stopped_mat.shape}"
            )
        out = np.zeros(m, dtype=float)
        for j in range(m):
            idx = np.where(stopped_mat[:, j])[0]
            if idx.size == 0:
                out[j] = 0.0
            else:
                out[j] = _macro_balanced_accuracy(preds_cube[idx, j, :], labels[idx])
        return out
    raise ValueError(f"Unknown utility: {utility}")


def compute_grid_pvalues(
    risks: np.ndarray,
    ns: np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> np.ndarray:
    """Per-lambda Hoeffding p-value for H0: R > alpha; risks shape (m,), ns shape (m,)."""
    m = risks.shape[0]
    out = np.ones(m, dtype=float)
    for j in range(m):
        n_j = int(ns[j])
        if n_j <= 0:
            out[j] = 1.0
            continue
        out[j] = hoeffding_p_value_mean_gt_alpha(float(risks[j]), n_j, float(alpha), B=float(B))
    return out


def compute_grid_pvalues_conditional_via_W(
    losses_mat: np.ndarray,
    stopped_mat: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-lambda p-values for H0: R_cond > alpha using the W-transform (full n).

    W_i^{(j)} = S_i^{(j)} (L_i^{(j)} - alpha). Under H0: R_cond > alpha we have
    E[W] > 0 at the boundary E[W] = 0; one-sided Hoeffding on mean(W) with
    W_i in [-alpha, 1-alpha] (length 1 when S=1) and 0 when S=0, so |W| <= max(alpha,1-alpha) <= 1.

    Returns (p_values, w_mean, n_vec) with n_vec[j] = n_total for all j (always).
    """
    losses_mat = np.asarray(losses_mat, dtype=float)
    stopped_mat = np.asarray(stopped_mat, dtype=float)
    n_total, m = losses_mat.shape
    alpha_f = float(alpha)
    p = np.ones(m, dtype=float)
    w_mean = np.zeros(m, dtype=float)
    n_vec = np.full(m, n_total, dtype=int)
    span = 1.0
    for j in range(m):
        w = stopped_mat[:, j] * (losses_mat[:, j] - alpha_f)
        w_hat = float(np.mean(w))
        w_mean[j] = w_hat
        p[j] = hoeffding_p_value_tail_mean_leq_c(
            w_hat, int(n_total), boundary_c=0.0, span_b_minus_a=float(span)
        )
    return p, w_mean, n_vec


def ltt_select_lambda(
    lambdas: np.ndarray,
    risks: np.ndarray,
    p_values: np.ndarray,
    delta: float,
    utilities: np.ndarray,
    default_lambda_idx: Optional[int] = None,
) -> Tuple[Optional[float], Dict]:
    """
    Apply Holm step-down on p_values at level delta, then choose argmax utility
    over the rejected (safe) lambdas.

    Returns (lambda_hat, meta). meta contains:
      - rejected: bool mask (m,)
      - safe_indices: list of indices in safe set
      - chosen_idx: index chosen (or None)
      - reason: str
      - trace: list of {j, lambda, r_hat, p_value, utility, rejected}
    """
    rejected = holm_step_down(p_values, delta)
    safe_indices = [int(i) for i in np.where(rejected)[0]]
    trace: List[Dict] = []
    for j, lam in enumerate(lambdas):
        trace.append(
            {
                "j": int(j),
                "lambda": float(lam),
                "r_hat": float(risks[j]),
                "p_value": float(p_values[j]),
                "utility": float(utilities[j]),
                "rejected": bool(rejected[j]),
            }
        )

    if not safe_indices:
        if default_lambda_idx is None:
            return None, {
                "rejected": rejected.tolist(),
                "safe_indices": safe_indices,
                "chosen_idx": None,
                "reason": "empty_safe_set",
                "trace": trace,
            }
        idx = int(default_lambda_idx)
        return float(lambdas[idx]), {
            "rejected": rejected.tolist(),
            "safe_indices": safe_indices,
            "chosen_idx": idx,
            "reason": "default_conservative",
            "trace": trace,
        }

    safe = np.array(safe_indices, dtype=int)
    safe_utilities = utilities[safe]
    best_local = int(np.argmax(safe_utilities))
    chosen_idx = int(safe[best_local])
    return float(lambdas[chosen_idx]), {
        "rejected": rejected.tolist(),
        "safe_indices": safe_indices,
        "chosen_idx": chosen_idx,
        "reason": "ltt_selected",
        "trace": trace,
    }
