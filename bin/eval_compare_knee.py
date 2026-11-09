#!/usr/bin/env python3
"""
FastMRI knee comparison: **LTT-conditional** (Holm), and forced baselines.

Mirrors eval_compare_cifar10.py for the knee_slice / knee_vol datasets.

**LTT-conditional (Option i):** for each λ, compute stopping times / predictions as before,
then per-label class-conditional binary FNR on stopped positives
``R_k = mean(ŷ_k=0 | stopped, y_k=1)`` (or FPR on stopped negatives). Exact binomial
p-values with ``p(λ)=max_k p_k``, then Holm at level δ and utility maximization.
Bonferroni (CASC) curves remain in JSON only (not drawn). Forced baselines unchanged.

By default the controlled risk is FNR; pass ``--risk_metric fpr`` for FPR, or
``--control_fnr_and_fpr`` to take the max p-value over all 2K components.
Reporting curves use the same class-conditional rates (max over labels), not the
legacy per-sample multilabel fraction loss. Pass ``--plot_conditional`` for the
middle panel on the cumulative stopped cohort only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_figure_export import (
    COMPARE_FIGSIZE_2COL,
    COMPARE_FIGSIZE_2x2,
    COMPARE_FIGSIZE_3COL,
    COMPARE_FIGSIZE_SINGLE,
    COMPARE_GRID_ALPHA,
    COMPARE_LEGEND_FONTSIZE,
    COMPARE_RCS_LEGEND_LABEL,
    COMPARE_XLABEL_SAMPLING,
    configure_compare_plot_style,
    draw_alpha_reference,
    plot_compare_series,
    print_compare_curve_table,
    print_label_prevalence,
    save_compare_figure,
    style_compare_ax,
)

from classifier.masks.cumulative_masking import CumulativeMaskState
from classifier.utils.utils import get_config
from eval_conformal_fnr_fpr import (
    FORCED_PRED_THRESHOLD,
    build_mask_for_sampling_rate,
    compute_dual_thresholds,
    get_probs_labels,
    load_model,
)
from eval_fst_knee import (
    add_ambiguity_band_args,
    build_dataloader,
    build_label_matrix,
    build_prob_matrix,
    build_score_cache_dir,
    build_uniform_alpha_schedule,
    collect_labels,
    compute_bands_from_argparse,
    compute_bounds,
    compute_preds_at_stop,
    compute_sample_fnr_loss,
    compute_sample_fpr_loss,
    compute_stop_indices,
    fnr_losses_with_abstain_zero,
    get_val_test_datasets,
    load_cached_scores,
    print_ambiguity_band_diagnostics,
    save_cached_scores,
    split_dataset,
    split_estimation_set,
    triage_band_config_dict,
)
from fst.ltt import (
    grid_class_conditional_pvalues,
    grid_utilities,
    ltt_select_lambda,
)

def forced_method_legend_label() -> str:
    """Legend for fixed-threshold forced stop; uses ``FORCED_PRED_THRESHOLD`` from ``eval_conformal_fnr_fpr``."""
    t = float(FORCED_PRED_THRESHOLD)
    return f"Forced ($p^+ > {t:g}$)"


def parse_alpha_per_label(raw: Optional[str]) -> Optional[np.ndarray]:
    """
    Parse ``--alpha_per_label`` as a comma-separated list of floats.

    Returns ``None`` when unset / empty (caller should broadcast scalar ``--alpha``).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    vals = np.asarray([float(p) for p in parts], dtype=float)
    if vals.ndim != 1 or vals.size < 1:
        raise ValueError(f"Could not parse --alpha_per_label={raw!r}")
    if np.any(~np.isfinite(vals)) or np.any(vals < 0.0) or np.any(vals > 1.0):
        raise ValueError(
            f"--alpha_per_label values must be finite and in [0, 1]; got {vals.tolist()}"
        )
    return vals


def resolve_alpha_per_label(
    alpha: float,
    alpha_per_label: Optional[np.ndarray],
    n_labels: int,
) -> np.ndarray:
    """Broadcast scalar ``alpha`` or validate length-``n_labels`` vector."""
    if alpha_per_label is None:
        return np.full(int(n_labels), float(alpha), dtype=float)
    a = np.asarray(alpha_per_label, dtype=float).ravel()
    if a.size != int(n_labels):
        raise ValueError(
            f"--alpha_per_label has length {a.size} but dataset has K={n_labels} labels"
        )
    return a


def resolve_control_label_indices(
    label_names: List[str],
    *,
    control_labels: Optional[str] = None,
    exclude_labels: Optional[str] = None,
) -> List[int]:
    """
    Resolve which label indices enter Option (i) ``p(λ)=max_k p_k``.

    Pass either ``--control_labels mtear,abnormal`` or ``--exclude_labels acl``
    (not both). Names are matched case-insensitively against ``label_names``.
    """
    names = [str(n).strip() for n in label_names]
    lower_to_idx = {n.lower(): i for i, n in enumerate(names)}

    def _parse_names(raw: Optional[str]) -> List[str]:
        if raw is None or not str(raw).strip():
            return []
        return [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]

    include = _parse_names(control_labels)
    exclude = _parse_names(exclude_labels)
    if include and exclude:
        raise ValueError("Pass only one of --control_labels / --exclude_labels")

    if include:
        idxs: List[int] = []
        for name in include:
            key = name.lower()
            if key not in lower_to_idx:
                raise ValueError(
                    f"--control_labels unknown label {name!r}; known={names}"
                )
            idxs.append(lower_to_idx[key])
        # preserve order, unique
        seen = set()
        out = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(i)
        if not out:
            raise ValueError("--control_labels resolved to an empty set")
        return out

    if exclude:
        excl = set()
        for name in exclude:
            key = name.lower()
            if key not in lower_to_idx:
                raise ValueError(
                    f"--exclude_labels unknown label {name!r}; known={names}"
                )
            excl.add(lower_to_idx[key])
        out = [i for i in range(len(names)) if i not in excl]
        if not out:
            raise ValueError("--exclude_labels removed every label")
        return out

    return list(range(len(names)))


def print_class_conditional_label_diagnostics(
    lambdas: np.ndarray,
    details: List[Dict[str, object]],
    label_names: List[str],
    *,
    delta: float,
    max_rows: int = 12,
) -> None:
    """
    Print per-label (n_k, R_k, p_k) for the λ with smallest p(λ), and how often
    each label is the argmax_p driver across the grid (blocker frequency).
    """
    m = len(details)
    if m == 0:
        print("Option (i) diagnostics: empty λ grid", flush=True)
        return
    p_all = np.asarray([float(d["p_value"]) for d in details], dtype=float)
    j_star = int(np.argmin(p_all))
    holm0 = float(delta) / max(m, 1)
    print(
        f"Option (i) per-label diagnostics (Holm tightest cutoff δ/m={holm0:.3g}):",
        flush=True,
    )

    # Driver frequency across λ
    driver_counts: Dict[int, int] = {}
    for d in details:
        lid = d.get("argmax_p_label_id")
        if lid is None:
            continue
        driver_counts[int(lid)] = driver_counts.get(int(lid), 0) + 1
    print("  How often each label has the max p_k across Λ (blocker frequency):", flush=True)
    for i, name in enumerate(label_names):
        c = int(driver_counts.get(i, 0))
        if c == 0 and i not in set(details[0].get("control_label_indices", range(len(label_names)))):
            continue
        print(f"    {name}: {c}/{m} ({100.0 * c / max(m, 1):.1f}%)", flush=True)

    d0 = details[j_star]
    print(
        f"  At λ with smallest p(λ)={float(d0['p_value']):.4g} "
        f"(λ={float(lambdas[j_star]):.4g}, j={j_star}):",
        flush=True,
    )
    print(
        f"    {'label':<12}{'kind':<5}{'n_k':>6}{'R_k':>8}{'p_k':>10}  driver?",
        flush=True,
    )
    argmax_c = int(d0.get("argmax_p_component", -1))
    for c, (lab, kind, n_k, r_k, p_k) in enumerate(
        zip(
            d0["label_ids"],
            d0["kinds"],
            d0["ns_per_component"],
            d0["risks_per_component"],
            d0["p_per_component"],
        )
    ):
        name = label_names[int(lab)] if int(lab) < len(label_names) else f"k{lab}"
        mark = "  <-- max p_k" if c == argmax_c else ""
        print(
            f"    {name:<12}{kind:<5}{int(n_k):>6}{float(r_k):>8.4f}{float(p_k):>10.3g}{mark}",
            flush=True,
        )

    # Also show a few λ with largest utility among those with p <= holm0 if any, else none
    safe_like = [j for j, d in enumerate(details) if float(d["p_value"]) <= holm0]
    if safe_like:
        print(
            f"  λ with p(λ)≤δ/m (necessary but not sufficient for Holm): {len(safe_like)}/{m}",
            flush=True,
        )
    else:
        print(
            "  No λ has p(λ)≤δ/m; none can enter the Holm safe set at the first step.",
            flush=True,
        )

    # Compact table of worst few λ by p (largest p) is less useful; show top-max_rows by min p
    order = np.argsort(p_all)[: max(1, int(max_rows))]
    print(
        f"  Smallest-{len(order)} p(λ) overview (driver label):",
        flush=True,
    )
    for j in order:
        d = details[int(j)]
        lid = d.get("argmax_p_label_id")
        dname = label_names[int(lid)] if lid is not None and int(lid) < len(label_names) else "?"
        print(
            f"    λ={float(lambdas[int(j)]):.4g}  p={float(d['p_value']):.4g}  "
            f"maxR={float(d['r_hat']):.4f}  min_n={int(d['n_used'])}  driver={dname}",
            flush=True,
        )


def cal_tuned_forced_legend_label(tuning_metric: str) -> str:
    """
    Legend for calibration-tuned θ per sampling rate.

    ``tuning_metric`` is kept for call-site compatibility (which objective picks θ on cal)
    but does not change the displayed string.
    """
    del tuning_metric
    return r"Forced ($p^+ > \theta^*$)"


def confidence_baseline_legend_label(threshold: float) -> str:
    return f"Confidence ≥ {float(threshold):g}"


def entropy_baseline_legend_label(threshold: float) -> str:
    return f"Entropy ≤ {float(threshold):g}"


# Backwards-compatible alias (eval_compare_knee_asym / prose); refreshed each import.
FORCED_METHOD_LABEL = forced_method_legend_label()

# Panel layout: tight_layout(rect=[0,0,1, TOP]) fits axes in y∈[0,TOP]. Legend with loc="lower center"
# must use the same y in bbox_to_anchor, else the figure top (1.0) vs TOP leaves a large gap.
FASTMRI_PANEL_LAYOUT_TOP = 0.88  # compact headroom for single-row legend

# Pretty-print names for per-label panel titles / y-labels.
LABEL_DISPLAY_NAMES = {
    "acl": "ACL",
    "mtear": "Meniscus",
    "abnormal": "Abnormal",
    "pirads_positive": r"PI-RADS $\geq$ 3",
}


def label_display_name(name: str) -> str:
    return LABEL_DISPLAY_NAMES.get(str(name), str(name))


METHOD_COLORS = {
    "ltt_conditional": "#8172B2",
    "nominal_bands": "#DD8452",
    "casc": "#55A868",
    "forced": "#7F7F7F",
    "cal_tuned_forced": "#B07C9E",
    "forced_fnr_cal": "#2CA02C",
    "confidence_threshold": "#BCBD22",
    "entropy_threshold": "#17BECF",
}

METHOD_LABELS = {
    "ltt_conditional": COMPARE_RCS_LEGEND_LABEL,
    "nominal_bands": "Nominal band (λ=1)",
    "casc": "Bonferroni",
    "forced": forced_method_legend_label(),
    "cal_tuned_forced": cal_tuned_forced_legend_label("balanced_accuracy"),
    "forced_fnr_cal": "Forced (cal θ, FNR cap)",
    "confidence_threshold": "Confidence threshold",
    "entropy_threshold": "Entropy threshold",
}

METHOD_LINESTYLES = {
    "ltt_conditional": "--",
    "nominal_bands": "-.",
    "casc": "-",
    "forced": "-",
    "cal_tuned_forced": "--",
    "forced_fnr_cal": "-.",
    "confidence_threshold": ":",
    "entropy_threshold": ":",
}

METHOD_MARKERS = {
    "ltt_conditional": "D",
    "nominal_bands": "^",
    "casc": "o",
    "forced": "o",
    "cal_tuned_forced": "s",
    "forced_fnr_cal": "^",
    "confidence_threshold": "v",
    "entropy_threshold": "^",
}

LAMBDA_METHODS = ["ltt_conditional"]
# Included on plots (not λ-selected on calibration): fixed dilation λ=1 → [b^-,\, b^+] from estimation.
# COMPARE_CURVE_METHODS = LAMBDA_METHODS + ["nominal_bands"]
COMPARE_CURVE_METHODS = list(LAMBDA_METHODS)


def fpr_losses_with_abstain_zero(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped: np.ndarray,
) -> np.ndarray:
    losses = np.zeros(labels.shape[0], dtype=float)
    if np.any(stopped):
        losses[stopped] = compute_sample_fpr_loss(preds[stopped], labels[stopped])
    return losses


def class_conditional_fnr_fpr_over_idx(
    preds: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    *,
    reduce: str = "max",
    label_indices: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """
    Class-conditional FNR / FPR on a sample subset ``idx`` (Option i reporting).

    For each controlled label k:
      FNR_k = mean(pred_k==0 | y_k==1) over samples in ``idx`` with y_k==1
      FPR_k = mean(pred_k==1 | y_k==0) over samples in ``idx`` with y_k==0
    Aggregate across labels with ``reduce`` in {``max``, ``mean``} over labels that
    have a positive denominator. Empty subset / no eligible labels → 0.

    ``label_indices`` restricts which labels enter the aggregate (default: all),
    matching Option (i) ``--exclude_labels`` / ``--control_labels``.
    """
    if idx.size == 0:
        return 0.0, 0.0
    p = preds[idx].astype(np.int64, copy=False)
    y = labels[idx].astype(np.int64, copy=False)
    n_labels = int(y.shape[1])
    labs = list(range(n_labels)) if label_indices is None else [int(i) for i in label_indices]
    fnrs: List[float] = []
    fprs: List[float] = []
    for k in labs:
        if k < 0 or k >= n_labels:
            raise ValueError(f"label_indices entry {k} out of range for K={n_labels}")
        pos = y[:, k] == 1
        n_pos = int(pos.sum())
        if n_pos > 0:
            fnrs.append(float(np.mean(p[pos, k] == 0)))
        neg = y[:, k] == 0
        n_neg = int(neg.sum())
        if n_neg > 0:
            fprs.append(float(np.mean(p[neg, k] == 1)))
    def _reduce(vals: List[float]) -> float:
        if not vals:
            return 0.0
        if reduce == "max":
            return float(np.max(vals))
        if reduce == "mean":
            return float(np.mean(vals))
        raise ValueError(f"Unknown reduce: {reduce}")
    return _reduce(fnrs), _reduce(fprs)


def class_conditional_fnr_fpr_unconditional(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped_mask: np.ndarray,
    *,
    reduce: str = "max",
    label_indices: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """
    Unconditional class-conditional rates: for each controlled label, numerator only
    counts errors among stopped samples; denominator is all positives / negatives in
    the full cohort (non-stopped units contribute 0 to the numerator).
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    stopped = np.asarray(stopped_mask, dtype=bool).ravel()
    n_labels = int(labels.shape[1])
    labs = list(range(n_labels)) if label_indices is None else [int(i) for i in label_indices]
    fnrs: List[float] = []
    fprs: List[float] = []
    for k in labs:
        if k < 0 or k >= n_labels:
            raise ValueError(f"label_indices entry {k} out of range for K={n_labels}")
        yk = labels[:, k].astype(np.int64, copy=False)
        pk = preds[:, k].astype(np.int64, copy=False)
        pos = yk == 1
        n_pos = int(pos.sum())
        if n_pos > 0:
            fn = int(np.sum(stopped & pos & (pk == 0)))
            fnrs.append(float(fn) / float(n_pos))
        neg = yk == 0
        n_neg = int(neg.sum())
        if n_neg > 0:
            fp = int(np.sum(stopped & neg & (pk == 1)))
            fprs.append(float(fp) / float(n_neg))
    def _reduce(vals: List[float]) -> float:
        if not vals:
            return 0.0
        if reduce == "max":
            return float(np.max(vals))
        if reduce == "mean":
            return float(np.mean(vals))
        raise ValueError(f"Unknown reduce: {reduce}")
    return _reduce(fnrs), _reduce(fprs)


def macro_fnr_fpr_over_idx(
    preds: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """
    Reporting FNR / FPR over subset ``idx``.

    Uses **class-conditional** per-label binary rates (max over controlled labels),
    matching Option (i) calibration — not the legacy per-sample multilabel fraction loss.
    """
    return class_conditional_fnr_fpr_over_idx(
        preds, labels, idx, reduce="max", label_indices=label_indices
    )


def resolve_controlled_label_names(
    label_names: Optional[Sequence[str]],
    label_indices: Optional[Sequence[int]],
    n_labels: int,
) -> Tuple[List[int], List[str]]:
    """Return ``(indices, names)`` for controlled labels used in per-label curves."""
    labs = list(range(n_labels)) if label_indices is None else [int(i) for i in label_indices]
    for k in labs:
        if k < 0 or k >= n_labels:
            raise ValueError(f"label_indices entry {k} out of range for K={n_labels}")
    if label_names is None:
        names = [f"label_{k}" for k in labs]
    else:
        if len(label_names) < n_labels:
            raise ValueError(
                f"label_names length {len(label_names)} < n_labels={n_labels}"
            )
        names = [str(label_names[k]) for k in labs]
    return labs, names


def class_conditional_fnr_fpr_per_label_over_idx(
    preds: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
]:
    """
    Per-label class-conditional FNR/FPR on subset ``idx`` (0.0 if empty denom).

    Also returns count maps ``(n_errors, n_denom)`` for FNR (FN among stopped
    positives) and FPR (FP among stopped negatives).
    """
    labs, names = resolve_controlled_label_names(
        label_names, label_indices, int(labels.shape[1])
    )
    fnr_map = {n: 0.0 for n in names}
    fpr_map = {n: 0.0 for n in names}
    fnr_counts = {n: (0, 0) for n in names}
    fpr_counts = {n: (0, 0) for n in names}
    if idx.size == 0:
        return fnr_map, fpr_map, fnr_counts, fpr_counts
    p = preds[idx].astype(np.int64, copy=False)
    y = labels[idx].astype(np.int64, copy=False)
    for k, name in zip(labs, names):
        pos = y[:, k] == 1
        n_pos = int(pos.sum())
        if n_pos > 0:
            n_fn = int(np.sum(p[pos, k] == 0))
            fnr_map[name] = float(n_fn) / float(n_pos)
            fnr_counts[name] = (n_fn, n_pos)
        neg = y[:, k] == 0
        n_neg = int(neg.sum())
        if n_neg > 0:
            n_fp = int(np.sum(p[neg, k] == 1))
            fpr_map[name] = float(n_fp) / float(n_neg)
            fpr_counts[name] = (n_fp, n_neg)
    return fnr_map, fpr_map, fnr_counts, fpr_counts


def class_conditional_fnr_fpr_per_label_unconditional(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped_mask: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Tuple[int, int]],
    Dict[str, Tuple[int, int]],
]:
    """
    Per-label unconditional class-conditional rates (non-stopped → 0 in numerator).

    Count maps use denom = all cohort positives/negatives for that label
    (``(n_err_among_stopped, n_label_total)``).
    """
    labs, names = resolve_controlled_label_names(
        label_names, label_indices, int(labels.shape[1])
    )
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    stopped = np.asarray(stopped_mask, dtype=bool).ravel()
    fnr_map = {n: 0.0 for n in names}
    fpr_map = {n: 0.0 for n in names}
    fnr_counts = {n: (0, 0) for n in names}
    fpr_counts = {n: (0, 0) for n in names}
    for k, name in zip(labs, names):
        yk = labels[:, k].astype(np.int64, copy=False)
        pk = preds[:, k].astype(np.int64, copy=False)
        pos = yk == 1
        n_pos = int(pos.sum())
        if n_pos > 0:
            fn = int(np.sum(stopped & pos & (pk == 0)))
            fnr_map[name] = float(fn) / float(n_pos)
            fnr_counts[name] = (fn, n_pos)
        neg = yk == 0
        n_neg = int(neg.sum())
        if n_neg > 0:
            fp = int(np.sum(stopped & neg & (pk == 1)))
            fpr_map[name] = float(fp) / float(n_neg)
            fpr_counts[name] = (fp, n_neg)
    return fnr_map, fpr_map, fnr_counts, fpr_counts


_PER_LABEL_RATE_KEYS = (
    "fnr_stopped_per_label",
    "fnr_cumulative_per_label",
    "fpr_stopped_per_label",
    "fpr_cumulative_per_label",
    "fnr_unconditional_cumulative_per_label",
    "fpr_unconditional_cumulative_per_label",
)

_PER_LABEL_COUNT_KEYS = (
    "fnr_stopped_counts_per_label",
    "fnr_cumulative_counts_per_label",
    "fpr_stopped_counts_per_label",
    "fpr_cumulative_counts_per_label",
    "fnr_unconditional_cumulative_counts_per_label",
    "fpr_unconditional_cumulative_counts_per_label",
)

_PER_LABEL_CURVE_KEYS = _PER_LABEL_RATE_KEYS + _PER_LABEL_COUNT_KEYS


def _empty_per_label_curve_block(
    names: Sequence[str], t_steps: int
) -> Dict[str, Dict[str, List[Any]]]:
    z_rate = [0.0] * int(t_steps)
    z_count = [(0, 0)] * int(t_steps)
    out: Dict[str, Dict[str, List[Any]]] = {
        key: {n: list(z_rate) for n in names} for key in _PER_LABEL_RATE_KEYS
    }
    for key in _PER_LABEL_COUNT_KEYS:
        out[key] = {n: list(z_count) for n in names}
    return out


def _new_per_label_accumulators(
    names: Sequence[str],
) -> Dict[str, Dict[str, List[Any]]]:
    return {key: {n: [] for n in names} for key in _PER_LABEL_CURVE_KEYS}


def _append_per_label_maps(
    acc: Dict[str, Dict[str, List[Any]]],
    key: str,
    values: Dict[str, Any],
    names: Sequence[str],
) -> None:
    for n in names:
        acc[key][n].append(values.get(n, 0.0 if "counts" not in key else (0, 0)))

def forced_fnr_cal_legend_label(
    *,
    alpha_ref: float,
    match_casc_schedule: bool,
    num_timesteps: int,
) -> str:
    """
    Legend for FN-constrained forced: per timestep and label θ is the largest value on an equi-spaced grid
    in [0,1] such that empirical FNR on calibration positives for that label is at most the stated cap.

    Cap is ``alpha_ref`` each timestep unless ``match_casc_schedule`` is True (then ``alpha_ref/T``).
    """
    t = max(int(num_timesteps), 1)
    if match_casc_schedule:
        cap = float(alpha_ref) / float(t)
        return f"Forced (cal θ | pos-FNR≤{cap:g}=α/{t})"
    return f"Forced (cal θ | pos-FNR≤{float(alpha_ref):g})"


def _empirical_fnr_on_positives_binary(pred: np.ndarray, y: np.ndarray) -> float:
    """Classical FNR on calibration units with y=1 for a single binary label column."""
    p = np.asarray(pred, dtype=np.int64, copy=False).ravel()
    yy = np.asarray(y, dtype=np.int64, copy=False).ravel()
    pos = yy == 1
    n_pos = int(np.sum(pos))
    if n_pos == 0:
        return 0.0
    fn = int(np.sum((p == 0) & pos))
    return float(fn / n_pos)


def best_threshold_binary_largest_theta_under_fnr_cap(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    fnr_cap: float,
    threshold_grid_pts: int,
) -> float:
    """
    Largest θ ∈ [0,1] grid such that empirical FNR on calibration positives ≤ ``fnr_cap``.

    Increasing θ lowers the positive prediction rate among true positives → FNR rises; feasible set is
    a prefix of an ascending θ grid ending at ``best_theta``.
    Vacuous positives (none on calibration for this label): return 1.0 (predict negative).
    """
    s = np.asarray(scores_cal, dtype=float).ravel()
    y = np.asarray(y_cal, dtype=np.int64).ravel()
    assert s.shape == y.shape
    if int(np.sum(y == 1)) == 0:
        return 1.0
    grid = np.linspace(0.0, 1.0, int(threshold_grid_pts))
    cap = float(fnr_cap)
    best_theta = 0.0
    tol = 1e-12
    for theta in grid:
        pred = (s >= theta).astype(np.int64)
        fnr_hat = _empirical_fnr_on_positives_binary(pred, y)
        if fnr_hat <= cap + tol:
            best_theta = float(theta)
    return best_theta


def _scalar_binary_calibration_objective(
    pred: np.ndarray,
    y: np.ndarray,
    tuning_metric: str,
) -> float:
    """Single-label scalar for threshold search on calibration (same BA convention as macro BA)."""
    p = pred.astype(np.int64, copy=False)
    yy = y.astype(np.int64, copy=False)
    tp = int(np.sum((p == 1) & (yy == 1)))
    tn = int(np.sum((p == 0) & (yy == 0)))
    fp = int(np.sum((p == 1) & (yy == 0)))
    fn = int(np.sum((p == 0) & (yy == 1)))
    if tuning_metric == "balanced_accuracy":
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.5 * (sens + spec)
    if tuning_metric == "accuracy":
        n = tp + tn + fp + fn
        return (tp + tn) / n if n > 0 else 0.0
    raise ValueError(f"Unknown tuning_metric: {tuning_metric}")


def best_threshold_binary_from_cal(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    tuning_metric: str,
    threshold_grid_pts: int,
) -> float:
    """
    Threshold θ ∈ [0,1] maximizing calibration objective independently for one label at one timestep.
    Ties broken by smallest θ among maximizers on the grid.
    """
    s = np.asarray(scores_cal, dtype=float).ravel()
    y = np.asarray(y_cal, dtype=np.int64).ravel()
    assert s.shape == y.shape
    grid = np.linspace(0.0, 1.0, int(threshold_grid_pts))
    best_theta = 0.5
    best_val = float("-inf")
    for theta in grid:
        pred = (s >= theta).astype(np.int64)
        val = _scalar_binary_calibration_objective(pred, y, tuning_metric)
        if val > best_val or (
            np.isclose(val, best_val) and theta < best_theta
        ):
            best_val = val
            best_theta = float(theta)
    return best_theta


def macro_balanced_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
) -> float:
    """
    Macro-average over labels of per-label balanced accuracy on subset `idx`.
    BA_k = (recall_k + specificity_k) / 2 when denominators positive; otherwise 0.
    """
    if idx.size == 0:
        return 0.0
    p = preds[idx].astype(np.int64, copy=False)
    y = labels[idx].astype(np.int64, copy=False)
    n_labels = int(y.shape[1])
    bas = []
    for k in range(n_labels):
        tp = int(np.sum((p[:, k] == 1) & (y[:, k] == 1)))
        tn = int(np.sum((p[:, k] == 0) & (y[:, k] == 0)))
        fp = int(np.sum((p[:, k] == 1) & (y[:, k] == 0)))
        fn = int(np.sum((p[:, k] == 0) & (y[:, k] == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bas.append(0.5 * (sens + spec))
    return float(np.mean(bas))


def per_lambda_calibration_arrays_knee_metric(
    pos_scores: np.ndarray,
    labels: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    lambda_candidates: np.ndarray,
    risk_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-λ calibration tensors for Option (i) / legacy diagnostics.

    Returns
    -------
    losses_mat : (n, m)
        Legacy per-sample FNR/FPR fraction losses (abstain → 0). Kept for fingerprints /
        JSON; LTT selection uses ``preds_cube`` instead.
    stopped_mat, stop_idx_mat : (n, m)
    preds_cube : (n, m, K)
        Binary predictions at each λ's stop time (undefined rows for non-stopped are
        still filled from ``compute_preds_at_stop`` but ignored via ``stopped_mat``).
    """
    n_cal = labels.shape[0]
    m = lambda_candidates.shape[0]
    n_labels = int(labels.shape[1])
    losses_mat = np.zeros((n_cal, m), dtype=float)
    stopped_mat = np.zeros((n_cal, m), dtype=bool)
    stop_idx_mat = np.zeros((n_cal, m), dtype=int)
    preds_cube = np.zeros((n_cal, m, n_labels), dtype=int)
    for j, lam in enumerate(lambda_candidates):
        lower, upper = compute_bounds(float(lam), b_pos, b_neg)
        stop_indices, stopped = compute_stop_indices(pos_scores, lower, upper)
        preds = compute_preds_at_stop(pos_scores, stop_indices, b_pos, b_neg)
        preds_cube[:, j, :] = preds
        if risk_metric == "fnr":
            losses_mat[:, j] = fnr_losses_with_abstain_zero(preds, labels, stopped)
        elif risk_metric == "fpr":
            losses_mat[:, j] = fpr_losses_with_abstain_zero(preds, labels, stopped)
        else:
            raise ValueError(f"Unknown risk_metric: {risk_metric}")
        stopped_mat[:, j] = stopped
        stop_idx_mat[:, j] = stop_indices
    return losses_mat, stopped_mat, stop_idx_mat, preds_cube


def select_lambda_methods(
    lambdas: np.ndarray,
    losses_mat: np.ndarray,
    stopped_mat: np.ndarray,
    stop_idx_mat: np.ndarray,
    alpha: float,
    delta: float,
    abstain_penalty: float,
    ltt_abstain_utility: str,
    ltt_conditional_utility: str,
    default_lambda_idx_abstain: Optional[int],
    default_lambda_idx_conditional: Optional[int],
    ltt_conditional_test: str = "via_W",
    *,
    preds_cube: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    risk_metric: str = "fnr",
    alpha_per_label: Optional[np.ndarray] = None,
    pvalue_method: str = "binomial",
    control_both: bool = False,
    control_label_indices: Optional[List[int]] = None,
) -> Dict[str, Dict]:
    """
    Returns {method: {"lambda_hat", "meta", "risks", "ns", "p_values", "utilities"}}.

    Only **LTT-conditional** is selected. When ``preds_cube`` and ``labels`` are given
    (required for FastMRI / prostate / hierarchical ImageNet), uses Option (i):
    per-label class-conditional binary FNR (or FPR) on stopped positives/negatives,
    binomial p-values, ``p(λ)=max_k p_k`` over ``control_label_indices`` (default: all).

    ``ltt_conditional_test`` (via_W / ratio) is ignored under Option (i); kept for CLI
    compatibility. ``abstain_penalty`` / abstain utility args are ignored.
    """
    del abstain_penalty, ltt_abstain_utility, default_lambda_idx_abstain
    if preds_cube is None or labels is None:
        raise ValueError(
            "select_lambda_methods requires preds_cube and labels for Option (i) "
            "class-conditional calibration"
        )

    alpha_spec: object = alpha if alpha_per_label is None else alpha_per_label
    p_cond, risks_cond, ns_cond, details = grid_class_conditional_pvalues(
        preds_cube,
        labels,
        stopped_mat,
        alpha_spec,
        risk_metric=risk_metric,
        pvalue_method=pvalue_method,
        control_both=control_both,
        control_label_indices=control_label_indices,
    )
    # losses_mat retained only so callers can fingerprint; not used for p-values.
    _ = losses_mat
    _ = ltt_conditional_test

    util_cond = grid_utilities(
        stopped_mat,
        stop_idx_mat,
        risks_cond,
        ltt_conditional_utility,
        preds_cube=preds_cube,
        labels=labels,
    )

    ltt_conditional_lam, ltt_conditional_meta = ltt_select_lambda(
        lambdas, risks_cond, p_cond, delta, util_cond,
        default_lambda_idx=default_lambda_idx_conditional,
    )
    ltt_conditional_meta = dict(ltt_conditional_meta)
    ltt_conditional_meta["class_conditional_details"] = details
    ltt_conditional_meta["control_label_indices"] = (
        None if control_label_indices is None else list(control_label_indices)
    )

    return {
        "ltt_conditional": {
            "lambda_hat": ltt_conditional_lam,
            "meta": ltt_conditional_meta,
            "risks": risks_cond.tolist(),
            "ns": ns_cond.tolist(),
            "p_values": p_cond.tolist(),
            "utilities": util_cond.tolist(),
            "w_mean": None,
            "conditional_test": "class_conditional_binomial",
            "pvalue_method": pvalue_method,
            "risk_metric": risk_metric,
            "control_both": bool(control_both),
            "control_label_indices": (
                None if control_label_indices is None else list(control_label_indices)
            ),
        },
    }


def lambda_method_curves(
    lambda_hat: Optional[float],
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Per-rate stopped count plus conditional / unconditional FNR / FPR curves."""
    n_test, t_steps, n_labels = test_pos_scores.shape
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    if lambda_hat is None:
        zeros_int = [0] * t_steps
        zeros_f = [0.0] * t_steps
        out = {
            "stopped_count": zeros_int,
            "fnr_stopped": list(zeros_f),
            "fnr_cumulative": list(zeros_f),
            "fpr_stopped": list(zeros_f),
            "fpr_cumulative": list(zeros_f),
            "fnr_unconditional_cumulative": list(zeros_f),
            "fpr_unconditional_cumulative": list(zeros_f),
            "balanced_accuracy_cumulative": list(zeros_f),
        }
        out.update(_empty_per_label_curve_block(ctrl_names, t_steps))
        return out

    lower, upper = compute_bounds(float(lambda_hat), b_pos, b_neg)
    stop_indices, stopped = compute_stop_indices(test_pos_scores, lower, upper)
    preds = compute_preds_at_stop(test_pos_scores, stop_indices, b_pos, b_neg)

    stopped_count: List[int] = []
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    per_lab = _new_per_label_accumulators(ctrl_names)
    for t in range(t_steps):
        idx_stopped = np.where(stopped & (stop_indices == t))[0]
        idx_cumulative = np.where(stopped & (stop_indices <= t))[0]
        stopped_count.append(int(idx_cumulative.size))

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(preds, label_matrix_test, idx_stopped, label_indices=label_indices)
        fnr_s.append(fnr_t)
        fpr_s.append(fpr_t)
        fnr_map_s, fpr_map_s, fnr_cnt_s, fpr_cnt_s = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, idx_stopped,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt_s, ctrl_names)

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(preds, label_matrix_test, idx_cumulative, label_indices=label_indices)
        fnr_c.append(fnr_t)
        fpr_c.append(fpr_t)
        fnr_map_c, fpr_map_c, fnr_cnt_c, fpr_cnt_c = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, idx_cumulative,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt_c, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(preds, label_matrix_test, idx_cumulative)
        )

        stopped_by_t = np.zeros(n_test, dtype=bool)
        stopped_by_t[idx_cumulative] = True
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            preds, label_matrix_test, stopped_by_t, reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            preds, label_matrix_test, stopped_by_t,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)

    out = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    out.update(per_lab)
    return out


def casc_curves(
    calib_pos_scores: np.ndarray,
    label_matrix_cal: np.ndarray,
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    alpha: float,
    positive_class_idx: int,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Per-rate dual-threshold (FNR/FPR) CASC, replicated from eval_fst_knee.py."""
    n_test, t_steps, n_labels = test_pos_scores.shape
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    alpha_k = build_uniform_alpha_schedule(t_steps, alpha)
    casc_active = np.ones(n_test, dtype=bool)
    casc_stop_indices = np.full(n_test, -1, dtype=int)
    casc_preds = np.zeros((n_test, n_labels), dtype=int)

    stopped_count: List[int] = []
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    per_lab = _new_per_label_accumulators(ctrl_names)

    for t in range(t_steps):
        pred_sets_by_label: List[List[List[int]]] = []
        for label_idx in range(n_labels):
            cal_probs_lbl = np.stack(
                [1.0 - calib_pos_scores[:, t, label_idx], calib_pos_scores[:, t, label_idx]],
                axis=1,
            )
            cal_labels_lbl = label_matrix_cal[:, label_idx].astype(int)
            thr = compute_dual_thresholds(
                cal_probs_lbl,
                cal_labels_lbl,
                alpha_fnr=float(alpha_k[t]),
                alpha_fpr=float(alpha_k[t]),
                pos_idx=positive_class_idx,
            )
            test_probs_lbl = np.stack(
                [1.0 - test_pos_scores[:, t, label_idx], test_pos_scores[:, t, label_idx]],
                axis=1,
            )
            sets = []
            for i in range(n_test):
                included = [k for k, threshold in thr.items() if test_probs_lbl[i, k] >= threshold]
                sets.append(included)
            pred_sets_by_label.append(sets)

        newly = []
        for i in np.where(casc_active)[0]:
            singleton_all = True
            pred_vec = np.zeros(n_labels, dtype=int)
            for label_idx in range(n_labels):
                s = pred_sets_by_label[label_idx][i]
                if len(s) != 1:
                    singleton_all = False
                    break
                pred_vec[label_idx] = 1 if s[0] == positive_class_idx else 0
            if singleton_all:
                newly.append((i, pred_vec))
        if newly:
            idxs = [i for i, _ in newly]
            casc_active[idxs] = False
            casc_stop_indices[idxs] = t
            for i, p in newly:
                casc_preds[i] = p

        idx_stopped = np.array([i for i, _ in newly], dtype=int)
        idx_cumulative = np.where(casc_stop_indices >= 0)[0]
        stopped_count.append(int(idx_cumulative.size))

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(casc_preds, label_matrix_test, idx_stopped, label_indices=label_indices)
        fnr_s.append(fnr_t)
        fpr_s.append(fpr_t)
        fnr_map_s, fpr_map_s, fnr_cnt_s, fpr_cnt_s = class_conditional_fnr_fpr_per_label_over_idx(
            casc_preds, label_matrix_test, idx_stopped,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt_s, ctrl_names)

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(casc_preds, label_matrix_test, idx_cumulative, label_indices=label_indices)
        fnr_c.append(fnr_t)
        fpr_c.append(fpr_t)
        fnr_map_c, fpr_map_c, fnr_cnt_c, fpr_cnt_c = class_conditional_fnr_fpr_per_label_over_idx(
            casc_preds, label_matrix_test, idx_cumulative,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt_c, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(casc_preds, label_matrix_test, idx_cumulative)
        )

        stopped_by_t = np.zeros(n_test, dtype=bool)
        stopped_by_t[idx_cumulative] = True
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            casc_preds, label_matrix_test, stopped_by_t, reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            casc_preds, label_matrix_test, stopped_by_t,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)

    out = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    out.update(per_lab)
    return out


def cal_tuned_forced_curves(
    calib_pos_scores: np.ndarray,
    label_matrix_cal: np.ndarray,
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    *,
    tuning_metric: str = "accuracy",
    threshold_grid_pts: int = 401,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    """
    Forced prediction at every rate, but per-label thresholds θ_{t,k} optimized on calibration
    at timestep t for the requested metric (same per-label BA / accuracy semantics as macros).
    """
    _, t_steps_cal, n_labels = calib_pos_scores.shape
    n_test, t_steps_test, n_labels_t = test_pos_scores.shape
    if n_labels != n_labels_t or t_steps_cal != t_steps_test:
        raise ValueError(
            f"Shape mismatch cal {calib_pos_scores.shape} vs test {test_pos_scores.shape}"
        )
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    thresholds = np.zeros((t_steps_test, n_labels), dtype=float)
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    stopped_count: List[int] = []
    per_lab = _new_per_label_accumulators(ctrl_names)
    all_idx = np.arange(n_test, dtype=int)
    for t in range(t_steps_test):
        theta_k = np.zeros(n_labels, dtype=float)
        for k in range(n_labels):
            theta_k[k] = best_threshold_binary_from_cal(
                calib_pos_scores[:, t, k],
                label_matrix_cal[:, k],
                tuning_metric=tuning_metric,
                threshold_grid_pts=threshold_grid_pts,
            )
        thresholds[t, :] = theta_k
        preds = (test_pos_scores[:, t, :] >= theta_k[np.newaxis, :]).astype(int)
        fnr_t, fpr_t = macro_fnr_fpr_over_idx(preds, label_matrix_test, all_idx, label_indices=label_indices)
        fnr_s.append(fnr_t)
        fnr_c.append(fnr_t)
        fpr_s.append(fpr_t)
        fpr_c.append(fpr_t)
        fnr_map, fpr_map, fnr_cnt, fpr_cnt = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, all_idx,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(preds, label_matrix_test, all_idx)
        )
        # All samples decide at this rate → unconditional == conditional class-cond.
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            preds, label_matrix_test, np.ones(n_test, dtype=bool), reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            preds, label_matrix_test, np.ones(n_test, dtype=bool),
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)
        stopped_count.append(n_test)
    curves = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    curves.update(per_lab)
    return curves, thresholds


def fnr_cap_constrained_forced_cal_curves(
    calib_pos_scores: np.ndarray,
    label_matrix_cal: np.ndarray,
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    *,
    fnr_cap_per_timestep: np.ndarray,
    threshold_grid_pts: int = 401,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    """
    Forced prediction at every rate: per (t, k) choose θ_{t,k} on calibration as the **largest** threshold
    on a uniform grid with empirical FNR on calibration positives for label k at most fnr_cap_per_timestep[t].
    """
    _, t_steps_cal, n_labels = calib_pos_scores.shape
    n_test, t_steps_test, n_labels_t = test_pos_scores.shape
    if n_labels != n_labels_t or t_steps_cal != t_steps_test:
        raise ValueError(
            f"Shape mismatch cal {calib_pos_scores.shape} vs test {test_pos_scores.shape}"
        )
    caps = np.asarray(fnr_cap_per_timestep, dtype=float).ravel()
    if caps.shape[0] != t_steps_test:
        raise ValueError(
            f"fnr_cap_per_timestep length {caps.shape[0]} != t_steps {t_steps_test}"
        )
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    thresholds = np.zeros((t_steps_test, n_labels), dtype=float)
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    stopped_count: List[int] = []
    per_lab = _new_per_label_accumulators(ctrl_names)
    all_idx = np.arange(n_test, dtype=int)
    for t in range(t_steps_test):
        theta_k = np.zeros(n_labels, dtype=float)
        cap_t = float(caps[t])
        for k in range(n_labels):
            theta_k[k] = best_threshold_binary_largest_theta_under_fnr_cap(
                calib_pos_scores[:, t, k],
                label_matrix_cal[:, k],
                fnr_cap=cap_t,
                threshold_grid_pts=threshold_grid_pts,
            )
        thresholds[t, :] = theta_k
        preds = (test_pos_scores[:, t, :] >= theta_k[np.newaxis, :]).astype(int)
        fnr_t, fpr_t = macro_fnr_fpr_over_idx(preds, label_matrix_test, all_idx, label_indices=label_indices)
        fnr_s.append(fnr_t)
        fnr_c.append(fnr_t)
        fpr_s.append(fpr_t)
        fpr_c.append(fpr_t)
        fnr_map, fpr_map, fnr_cnt, fpr_cnt = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, all_idx,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(preds, label_matrix_test, all_idx)
        )
        # All samples decide at this rate → unconditional == conditional class-cond.
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            preds, label_matrix_test, np.ones(n_test, dtype=bool), reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            preds, label_matrix_test, np.ones(n_test, dtype=bool),
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)
        stopped_count.append(n_test)
    curves = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    curves.update(per_lab)
    return curves, thresholds


def forced_curves(
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """Per-rate Forced metrics: every sample 'decides' at every rate."""
    n_test, t_steps, n_labels = test_pos_scores.shape
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    stopped_count: List[int] = []
    per_lab = _new_per_label_accumulators(ctrl_names)
    all_idx = np.arange(test_pos_scores.shape[0], dtype=int)
    for t in range(t_steps):
        forced_preds = (test_pos_scores[:, t, :] >= FORCED_PRED_THRESHOLD).astype(int)
        fnr_t, fpr_t = macro_fnr_fpr_over_idx(forced_preds, label_matrix_test, all_idx, label_indices=label_indices)
        fnr_s.append(fnr_t)
        fnr_c.append(fnr_t)
        fpr_s.append(fpr_t)
        fpr_c.append(fpr_t)
        fnr_map, fpr_map, fnr_cnt, fpr_cnt = class_conditional_fnr_fpr_per_label_over_idx(
            forced_preds, label_matrix_test, all_idx,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(forced_preds, label_matrix_test, all_idx)
        )
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            forced_preds, label_matrix_test, np.ones(n_test, dtype=bool), reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            forced_preds, label_matrix_test, np.ones(n_test, dtype=bool),
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)
        stopped_count.append(n_test)
    out = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    out.update(per_lab)
    return out


def _first_stop_indices_confidence_all_labels(
    test_pos_scores: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    """First timestep where every label's positive score is >= ``confidence_threshold`` (else -1)."""
    n_test, t_steps, _n_lab = test_pos_scores.shape
    first = np.full(n_test, -1, dtype=int)
    thr = float(confidence_threshold)
    for t in range(t_steps):
        ok = np.all(test_pos_scores[:, t, :] >= thr, axis=1)
        m = (first < 0) & ok
        first[m] = t
    return first


def _first_stop_indices_mean_label_entropy(
    test_pos_scores: np.ndarray,
    entropy_threshold: float,
) -> np.ndarray:
    """First timestep where mean binary (per-label) entropy is <= ``entropy_threshold``."""
    n_test, t_steps, _n_lab = test_pos_scores.shape
    first = np.full(n_test, -1, dtype=int)
    thr = float(entropy_threshold)
    for t in range(t_steps):
        p = np.clip(test_pos_scores[:, t, :], 1e-12, 1.0 - 1e-12)
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).mean(axis=1)
        ok = h <= thr
        m = (first < 0) & ok
        first[m] = t
    return first


def _curves_from_first_stop_indices(
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    stop_indices: np.ndarray,
    *,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, List[float]]:
    """
    Same per-rate metrics as ``lambda_method_curves`` but with fixed first-stop indices.
    Predictions at stop use ``FORCED_PRED_THRESHOLD`` on positive scores (binary per label).
    """
    n_test, t_steps, n_labels = test_pos_scores.shape
    _, ctrl_names = resolve_controlled_label_names(label_names, label_indices, n_labels)
    stopped = stop_indices >= 0
    preds = np.zeros((n_test, n_labels), dtype=int)
    if np.any(stopped):
        ii = np.where(stopped)[0]
        ts = stop_indices[ii]
        for lbl in range(n_labels):
            preds[ii, lbl] = (
                test_pos_scores[ii, ts, lbl] >= FORCED_PRED_THRESHOLD
            ).astype(int)

    stopped_count: List[int] = []
    fnr_s, fnr_c, fpr_s, fpr_c = [], [], [], []
    fnr_uc, fpr_uc = [], []
    balanced_acc_cum: List[float] = []
    per_lab = _new_per_label_accumulators(ctrl_names)
    for t in range(t_steps):
        idx_stopped = np.where(stopped & (stop_indices == t))[0]
        idx_cumulative = np.where(stopped & (stop_indices <= t))[0]
        stopped_count.append(int(idx_cumulative.size))

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(
            preds, label_matrix_test, idx_stopped, label_indices=label_indices
        )
        fnr_s.append(fnr_t)
        fpr_s.append(fpr_t)
        fnr_map_s, fpr_map_s, fnr_cnt_s, fpr_cnt_s = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, idx_stopped,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_stopped_per_label", fnr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_per_label", fpr_map_s, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_stopped_counts_per_label", fnr_cnt_s, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_stopped_counts_per_label", fpr_cnt_s, ctrl_names)

        fnr_t, fpr_t = macro_fnr_fpr_over_idx(
            preds, label_matrix_test, idx_cumulative, label_indices=label_indices
        )
        fnr_c.append(fnr_t)
        fpr_c.append(fpr_t)
        fnr_map_c, fpr_map_c, fnr_cnt_c, fpr_cnt_c = class_conditional_fnr_fpr_per_label_over_idx(
            preds, label_matrix_test, idx_cumulative,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_cumulative_per_label", fnr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_per_label", fpr_map_c, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_cumulative_counts_per_label", fnr_cnt_c, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_cumulative_counts_per_label", fpr_cnt_c, ctrl_names)
        balanced_acc_cum.append(
            macro_balanced_accuracy(preds, label_matrix_test, idx_cumulative)
        )

        stopped_by_t = np.zeros(n_test, dtype=bool)
        stopped_by_t[idx_cumulative] = True
        fnr_u, fpr_u = class_conditional_fnr_fpr_unconditional(
            preds, label_matrix_test, stopped_by_t, reduce="max", label_indices=label_indices
        )
        fnr_uc.append(fnr_u)
        fpr_uc.append(fpr_u)
        fnr_map_u, fpr_map_u, fnr_cnt_u, fpr_cnt_u = class_conditional_fnr_fpr_per_label_unconditional(
            preds, label_matrix_test, stopped_by_t,
            label_indices=label_indices, label_names=label_names,
        )
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_per_label", fnr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_per_label", fpr_map_u, ctrl_names)
        _append_per_label_maps(per_lab, "fnr_unconditional_cumulative_counts_per_label", fnr_cnt_u, ctrl_names)
        _append_per_label_maps(per_lab, "fpr_unconditional_cumulative_counts_per_label", fpr_cnt_u, ctrl_names)

    out = {
        "stopped_count": stopped_count,
        "fnr_stopped": fnr_s,
        "fnr_cumulative": fnr_c,
        "fpr_stopped": fpr_s,
        "fpr_cumulative": fpr_c,
        "fnr_unconditional_cumulative": fnr_uc,
        "fpr_unconditional_cumulative": fpr_uc,
        "balanced_accuracy_cumulative": balanced_acc_cum,
    }
    out.update(per_lab)
    return out


def fixed_threshold_schedule_curves(
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    *,
    confidence_threshold: Optional[float] = None,
    entropy_threshold: Optional[float] = None,
    label_indices: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Dict[str, List[float]]], Optional[Dict[str, List[float]]]]:
    """
    Non-λ baselines: stop each sample at the first sampling rate (timestep) where either

    * **confidence**: all per-label positive scores are >= ``confidence_threshold``, or
    * **entropy**: mean per-label binary entropy is <= ``entropy_threshold``.

    Mirrors ``eval_conformal_fnr_fpr`` / ``eval_conformal_coverage`` threshold semantics
    (binary heads on ``[0, 1]`` positive scores).
    """
    conf_curve: Optional[Dict[str, List[float]]] = None
    ent_curve: Optional[Dict[str, List[float]]] = None
    if confidence_threshold is not None:
        si = _first_stop_indices_confidence_all_labels(
            test_pos_scores, float(confidence_threshold)
        )
        conf_curve = _curves_from_first_stop_indices(
            test_pos_scores, label_matrix_test, si,
            label_indices=label_indices, label_names=label_names,
        )
    if entropy_threshold is not None:
        si = _first_stop_indices_mean_label_entropy(test_pos_scores, float(entropy_threshold))
        ent_curve = _curves_from_first_stop_indices(
            test_pos_scores, label_matrix_test, si,
            label_indices=label_indices, label_names=label_names,
        )
    return conf_curve, ent_curve


def plot_compare_stopped(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    file_tag: str,
    n_test: int,
    *,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    """Fraction stopped vs sampling rate (LTT-conditional and optional baselines)."""
    if n_test <= 0:
        raise ValueError("n_test must be positive for stopped fraction plots")
    inv = 1.0 / float(n_test)
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)
    for key in COMPARE_CURVE_METHODS:
        frac = [int(c) * inv for c in curves_by_method[key]["stopped_count"]]
        ax.plot(
            sampling_rates,
            frac,
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        frac_ct = [int(c) * inv for c in cal_tuned_forced_curve["stopped_count"]]
        ax.plot(
            sampling_rates,
            frac_ct,
            marker=METHOD_MARKERS["cal_tuned_forced"],
            linestyle=METHOD_LINESTYLES["cal_tuned_forced"],
            color=METHOD_COLORS["cal_tuned_forced"],
            label=label,
        )
    if forced_fnr_cal_curve is not None:
        label_f = forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"]
        frac_f = [int(c) * inv for c in forced_fnr_cal_curve["stopped_count"]]
        ax.plot(
            sampling_rates,
            frac_f,
            marker=METHOD_MARKERS["forced_fnr_cal"],
            linestyle=METHOD_LINESTYLES["forced_fnr_cal"],
            color=METHOD_COLORS["forced_fnr_cal"],
            label=label_f,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        frac_c = [int(c) * inv for c in confidence_threshold_curve["stopped_count"]]
        ax.plot(
            sampling_rates,
            frac_c,
            marker=METHOD_MARKERS["confidence_threshold"],
            linestyle=METHOD_LINESTYLES["confidence_threshold"],
            color=METHOD_COLORS["confidence_threshold"],
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        frac_e = [int(c) * inv for c in entropy_threshold_curve["stopped_count"]]
        ax.plot(
            sampling_rates,
            frac_e,
            marker=METHOD_MARKERS["entropy_threshold"],
            linestyle=METHOD_LINESTYLES["entropy_threshold"],
            color=METHOD_COLORS["entropy_threshold"],
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel("Fraction of test samples stopped")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=COMPARE_GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_stopped_vs_sampling_rate_{file_tag}.png"
    save_compare_figure(fig, out_path)


def _metric_series_max_y(
    metric_key: str,
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    *,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
) -> float:
    vals: List[float] = []
    for key in COMPARE_CURVE_METHODS:
        vals.extend(float(x) for x in curves_by_method[key][metric_key])
    vals.extend(float(x) for x in forced_curve[metric_key])
    if cal_tuned_forced_curve is not None:
        vals.extend(float(x) for x in cal_tuned_forced_curve[metric_key])
    if forced_fnr_cal_curve is not None:
        vals.extend(float(x) for x in forced_fnr_cal_curve[metric_key])
    if confidence_threshold_curve is not None:
        vals.extend(float(x) for x in confidence_threshold_curve[metric_key])
    if entropy_threshold_curve is not None:
        vals.extend(float(x) for x in entropy_threshold_curve[metric_key])
    if not vals:
        return 0.0
    m = float(np.nanmax(vals))
    return m if np.isfinite(m) else 0.0


def _metric_series_min_y(
    metric_key: str,
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    *,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
) -> float:
    vals: List[float] = []
    for key in COMPARE_CURVE_METHODS:
        vals.extend(float(x) for x in curves_by_method[key][metric_key])
    vals.extend(float(x) for x in forced_curve[metric_key])
    if cal_tuned_forced_curve is not None:
        vals.extend(float(x) for x in cal_tuned_forced_curve[metric_key])
    if forced_fnr_cal_curve is not None:
        vals.extend(float(x) for x in forced_fnr_cal_curve[metric_key])
    if confidence_threshold_curve is not None:
        vals.extend(float(x) for x in confidence_threshold_curve[metric_key])
    if entropy_threshold_curve is not None:
        vals.extend(float(x) for x in entropy_threshold_curve[metric_key])
    if not vals:
        return 0.0
    m = float(np.nanmin(vals))
    return m if np.isfinite(m) else 0.0


def _cumulative_risk_ylim(
    metric_key: str,
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    *,
    aligned: bool,
    control_alpha: Optional[float],
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
) -> Tuple[float, float]:
    y_top = _metric_series_max_y(
        metric_key,
        curves_by_method,
        casc_curve,
        forced_curve,
        cal_tuned_forced_curve=cal_tuned_forced_curve,
        forced_fnr_cal_curve=forced_fnr_cal_curve,
        confidence_threshold_curve=confidence_threshold_curve,
        entropy_threshold_curve=entropy_threshold_curve,
    )
    y_bot = _metric_series_min_y(
        metric_key,
        curves_by_method,
        casc_curve,
        forced_curve,
        cal_tuned_forced_curve=cal_tuned_forced_curve,
        forced_fnr_cal_curve=forced_fnr_cal_curve,
        confidence_threshold_curve=confidence_threshold_curve,
        entropy_threshold_curve=entropy_threshold_curve,
    )
    if aligned and control_alpha is not None:
        ca = float(control_alpha)
        y_top = max(y_top, ca)
        y_bot = min(y_bot, ca)
    pad_abs = 0.02
    span = max(y_top - y_bot, 1e-12)
    pad = max(pad_abs, 0.05 * span)
    y_min = max(0.0, y_bot - pad)
    y_max = min(1.05, y_top + pad)
    if y_max <= y_min:
        y_max = min(1.05, y_min + 0.05)
    return y_min, y_max


def _merge_legend_by_priority(
    axes: List[Axes],
    label_priority: List[str],
) -> Tuple[List, List]:
    """Stable legend: first occurrence of each label wins; order follows ``label_priority`` then leftovers."""
    by_label: Dict[str, object] = {}
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for h, lab in zip(hs, ls):
            if lab not in by_label:
                by_label[lab] = h
    ordered_h, ordered_l = [], []
    for lab in label_priority:
        if lab in by_label:
            ordered_h.append(by_label[lab])
            ordered_l.append(lab)
    for lab, h in by_label.items():
        if lab not in ordered_l:
            ordered_h.append(h)
            ordered_l.append(lab)
    return ordered_h, ordered_l


def _fastmri_legend_priority(
    cal_tuned_forced_legend: Optional[str],
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> List[str]:
    p = [METHOD_LABELS[k] for k in COMPARE_CURVE_METHODS]
    p.append(forced_method_legend_label())
    p.append(cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"])
    if forced_fnr_cal_legend:
        p.append(forced_fnr_cal_legend)
    if confidence_threshold_legend:
        p.append(confidence_threshold_legend)
    if entropy_threshold_legend:
        p.append(entropy_threshold_legend)
    return p


def _style_compare_ax(ax: Axes, *, grid_y: bool = True) -> None:
    """Publication styling: soft face, despine, light grid (shared with α-sweep)."""
    style_compare_ax(ax, grid=True, grid_y=grid_y)


def _plot_series(
    ax: Axes,
    sampling_rates: Sequence[float],
    values: Sequence[float],
    *,
    method_key: str,
    label: str,
    zorder: int = 2,
) -> None:
    plot_compare_series(
        ax,
        sampling_rates,
        values,
        color=METHOD_COLORS[method_key],
        linestyle=METHOD_LINESTYLES[method_key],
        marker=METHOD_MARKERS[method_key],
        label=label,
        zorder=zorder,
    )


def _panel_risk_ylim(
    metric_key: str,
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    *,
    control_alpha: Optional[float],
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
) -> Tuple[float, float]:
    """
    Readable FNR/FPR y-limits around α.

    Forced (p⁺>0.5) can sit far above α early on; keep a band that shows α,
    RCS, and calibrated/heuristic baselines clearly. Forced may clip at the top.
    """
    del casc_curve
    primary: List[float] = []
    for key in COMPARE_CURVE_METHODS:
        primary.extend(float(x) for x in curves_by_method[key][metric_key])
    for curve in (
        cal_tuned_forced_curve,
        forced_fnr_cal_curve,
        confidence_threshold_curve,
        entropy_threshold_curve,
    ):
        if curve is not None:
            primary.extend(float(x) for x in curve[metric_key])
    primary = [v for v in primary if np.isfinite(v)]
    forced_vals = [float(x) for x in forced_curve[metric_key] if np.isfinite(float(x))]
    y_pri = max(primary) if primary else 0.0
    y_f = max(forced_vals) if forced_vals else 0.0
    ca = float(control_alpha) if control_alpha is not None else 0.0
    y_top = max(y_pri, y_f, ca, 0.08)
    y_top = max(y_top * 1.08, ca * 1.5 if ca > 0 else y_top)
    return 0.0, min(1.05, y_top)


def _draw_stopped_fraction_ax(
    ax: Axes,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    n_test: int,
    *,
    forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    del casc_curve  # reserved for optional Bonferroni overlay
    inv = 1.0 / float(n_test)
    for key in COMPARE_CURVE_METHODS:
        frac = [int(c) * inv for c in curves_by_method[key]["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac,
            method_key=key,
            label=METHOD_LABELS[key],
            zorder=3,
        )
    # Forced always evaluates every sample at each rate → fraction stopped = 1.
    if forced_curve is not None:
        frac_forced = [int(c) * inv for c in forced_curve["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac_forced,
            method_key="forced",
            label=METHOD_LABELS["forced"],
            zorder=1,
        )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        frac_ct = [int(c) * inv for c in cal_tuned_forced_curve["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac_ct,
            method_key="cal_tuned_forced",
            label=label,
            zorder=2,
        )
    if forced_fnr_cal_curve is not None:
        label_f = forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"]
        frac_f = [int(c) * inv for c in forced_fnr_cal_curve["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac_f,
            method_key="forced_fnr_cal",
            label=label_f,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        frac_c = [int(c) * inv for c in confidence_threshold_curve["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac_c,
            method_key="confidence_threshold",
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        frac_e = [int(c) * inv for c in entropy_threshold_curve["stopped_count"]]
        _plot_series(
            ax,
            sampling_rates,
            frac_e,
            method_key="entropy_threshold",
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel("Fraction stopped")
    ax.set_ylim(0.0, 1.05)
    _style_compare_ax(ax)


def _draw_controlled_risk_ax(
    ax: Axes,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    metric_key: str,
    metric_ylabel: str,
    *,
    control_alpha: Optional[float] = None,
    risk_metric: Optional[str] = None,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    aligned = False
    if control_alpha is not None and risk_metric is not None:
        if risk_metric == "fnr" and metric_key in (
            "fnr_cumulative",
            "fnr_unconditional_cumulative",
            "_panel_metric",
        ):
            aligned = True
        elif risk_metric == "fpr" and metric_key in (
            "fpr_cumulative",
            "fpr_unconditional_cumulative",
            "_panel_metric",
        ):
            aligned = True
    if aligned:
        draw_alpha_reference(ax, float(control_alpha))
    for key in COMPARE_CURVE_METHODS:
        _plot_series(
            ax,
            sampling_rates,
            curves_by_method[key][metric_key],
            method_key=key,
            label=METHOD_LABELS[key],
            zorder=3,
        )
    _plot_series(
        ax,
        sampling_rates,
        forced_curve[metric_key],
        method_key="forced",
        label=METHOD_LABELS["forced"],
        zorder=1,
    )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        _plot_series(
            ax,
            sampling_rates,
            cal_tuned_forced_curve[metric_key],
            method_key="cal_tuned_forced",
            label=label,
        )
    if forced_fnr_cal_curve is not None:
        label_f = forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"]
        _plot_series(
            ax,
            sampling_rates,
            forced_fnr_cal_curve[metric_key],
            method_key="forced_fnr_cal",
            label=label_f,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        _plot_series(
            ax,
            sampling_rates,
            confidence_threshold_curve[metric_key],
            method_key="confidence_threshold",
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        _plot_series(
            ax,
            sampling_rates,
            entropy_threshold_curve[metric_key],
            method_key="entropy_threshold",
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel(metric_ylabel)
    _style_compare_ax(ax)
    if metric_key in (
        "fnr_cumulative",
        "fnr_unconditional_cumulative",
        "fpr_cumulative",
        "fpr_unconditional_cumulative",
        "_panel_metric",
    ):
        if metric_key == "_panel_metric" and aligned:
            ymin, ymax = _panel_risk_ylim(
                metric_key,
                curves_by_method,
                casc_curve,
                forced_curve,
                control_alpha=control_alpha,
                cal_tuned_forced_curve=cal_tuned_forced_curve,
                forced_fnr_cal_curve=forced_fnr_cal_curve,
                confidence_threshold_curve=confidence_threshold_curve,
                entropy_threshold_curve=entropy_threshold_curve,
            )
        else:
            ymin, ymax = _cumulative_risk_ylim(
                metric_key,
                curves_by_method,
                casc_curve,
                forced_curve,
                aligned=aligned,
                control_alpha=control_alpha,
                cal_tuned_forced_curve=cal_tuned_forced_curve,
                forced_fnr_cal_curve=forced_fnr_cal_curve,
                confidence_threshold_curve=confidence_threshold_curve,
                entropy_threshold_curve=entropy_threshold_curve,
            )
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.05)


def _draw_balanced_accuracy_ax(
    ax: Axes,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    *,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    metric_key = "balanced_accuracy_cumulative"
    for key in COMPARE_CURVE_METHODS:
        _plot_series(
            ax,
            sampling_rates,
            curves_by_method[key][metric_key],
            method_key=key,
            label=METHOD_LABELS[key],
            zorder=3,
        )
    _plot_series(
        ax,
        sampling_rates,
        forced_curve[metric_key],
        method_key="forced",
        label=METHOD_LABELS["forced"],
        zorder=1,
    )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        _plot_series(
            ax,
            sampling_rates,
            cal_tuned_forced_curve[metric_key],
            method_key="cal_tuned_forced",
            label=label,
        )
    if forced_fnr_cal_curve is not None:
        label_f = forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"]
        _plot_series(
            ax,
            sampling_rates,
            forced_fnr_cal_curve[metric_key],
            method_key="forced_fnr_cal",
            label=label_f,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        _plot_series(
            ax,
            sampling_rates,
            confidence_threshold_curve[metric_key],
            method_key="confidence_threshold",
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        _plot_series(
            ax,
            sampling_rates,
            entropy_threshold_curve[metric_key],
            method_key="entropy_threshold",
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel("Cumulative balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    _style_compare_ax(ax)


def _project_per_label_metric_curves(
    curves: Dict[str, Any],
    nested_key: str,
    label_name: str,
    flat_key: str = "_panel_metric",
) -> Dict[str, Any]:
    """Copy a curve dict with one per-label series exposed under ``flat_key``."""
    out = dict(curves)
    nested = curves.get(nested_key)
    if not isinstance(nested, dict) or label_name not in nested:
        raise KeyError(
            f"curve missing per-label series {nested_key!r}[{label_name!r}]"
        )
    out[flat_key] = nested[label_name]
    return out


def _project_method_curves_per_label(
    curves_by_method: Dict[str, Dict[str, Any]],
    nested_key: str,
    label_name: str,
    flat_key: str = "_panel_metric",
) -> Dict[str, Dict[str, Any]]:
    return {
        k: _project_per_label_metric_curves(v, nested_key, label_name, flat_key)
        for k, v in curves_by_method.items()
    }


def plot_compare_fastmri_panel(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    file_tag: str,
    n_test: int,
    *,
    risk_metric: str,
    control_alpha: float,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
    plot_conditional_middle: bool = False,
    control_label_names: Optional[Sequence[str]] = None,
    control_alpha_by_label: Optional[Dict[str, float]] = None,
    label_prevalence_rows: Optional[Sequence[Tuple[str, int, int]]] = None,
    label_prevalence_split: str = "test",
) -> None:
    """
    FastMRI compare panel.

    When ``control_label_names`` is provided (knee / multilabel Option i), draws a
    2×2 grid: fraction stopped, stopped FNR/FPR for each controlled label (up to two
    in the middle/bottom), and balanced accuracy filling the remaining cell when
    there are ≤2 controlled labels. With three controlled labels the three risk
    panels occupy the remaining 2×2 cells (no BA).

    Without ``control_label_names``, keeps the legacy 1×3 layout (aggregate max-over-
    labels risk + BA).
    """
    if n_test <= 0:
        raise ValueError("n_test must be positive for stopped fraction plots")
    if risk_metric not in ("fnr", "fpr"):
        raise ValueError(f"Unknown risk_metric for panel: {risk_metric}")

    ctrl_names = [str(n) for n in control_label_names] if control_label_names else []
    # Per-label 2×2 only when ≥2 controlled labels (knee ACL+meniscus). Binary tasks
    # (prostate) keep the publication 1×3 layout with a single risk panel.
    use_per_label = len(ctrl_names) > 1

    if label_prevalence_rows:
        print_label_prevalence(
            label_prevalence_rows,
            split_name=label_prevalence_split,
            controlled_names=ctrl_names if ctrl_names else None,
        )

    # Per-label panels always show stopped-cohort class-conditional rates.
    if risk_metric == "fnr":
        nested_key = "fnr_cumulative_per_label"
        risk_word = "FNR"
        legacy_key = "fnr_cumulative" if plot_conditional_middle else "fnr_unconditional_cumulative"
        if plot_conditional_middle and len(ctrl_names) == 1:
            legacy_ylabel = f"Stopped FNR ({label_display_name(ctrl_names[0])})"
        elif plot_conditional_middle:
            legacy_ylabel = "Cumulative class-cond. FNR (stopped)"
        else:
            legacy_ylabel = "Cumulative class-cond. FNR"
    else:
        nested_key = "fpr_cumulative_per_label"
        risk_word = "FPR"
        legacy_key = "fpr_cumulative" if plot_conditional_middle else "fpr_unconditional_cumulative"
        if plot_conditional_middle and len(ctrl_names) == 1:
            legacy_ylabel = f"Stopped FPR ({label_display_name(ctrl_names[0])})"
        elif plot_conditional_middle:
            legacy_ylabel = "Cumulative class-cond. FPR (stopped)"
        else:
            legacy_ylabel = "Cumulative class-cond. FPR"

    table_rows: List[Tuple[str, Dict[str, List[float]]]] = [
        (METHOD_LABELS[key], curves_by_method[key]) for key in COMPARE_CURVE_METHODS
    ]
    table_rows.append((METHOD_LABELS["forced"], forced_curve))
    if cal_tuned_forced_curve is not None:
        table_rows.append(
            (
                cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"],
                cal_tuned_forced_curve,
            )
        )
    if forced_fnr_cal_curve is not None:
        table_rows.append(
            (
                forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"],
                forced_fnr_cal_curve,
            )
        )
    if confidence_threshold_curve is not None:
        table_rows.append(
            (
                confidence_threshold_legend or METHOD_LABELS["confidence_threshold"],
                confidence_threshold_curve,
            )
        )
    if entropy_threshold_curve is not None:
        table_rows.append(
            (
                entropy_threshold_legend or METHOD_LABELS["entropy_threshold"],
                entropy_threshold_curve,
            )
        )

    if use_per_label:
        metric_keys = [f"{nested_key}.{n}" for n in ctrl_names]
        metric_labels = [f"stopped {risk_word}_{n}" for n in ctrl_names]
        nested_count_key = nested_key.replace("_per_label", "_counts_per_label")
        metric_count_keys = [f"{nested_count_key}.{n}" for n in ctrl_names]
        print_compare_curve_table(
            sampling_rates,
            table_rows,
            metric_key=metric_keys,
            metric_label=metric_labels,
            metric_count_key=metric_count_keys,
            title=(
                f"Plot data table [{file_tag}]  "
                f"(cell = (stopped_count, {', '.join(metric_labels)} "
                f"each as rate (n_err/n_pos_stopped)))"
            ),
        )
    else:
        print_compare_curve_table(
            sampling_rates,
            table_rows,
            metric_key=legacy_key,
            metric_label=legacy_ylabel,
            title=(
                f"Plot data table [{file_tag}"
                f"{'_conditional' if plot_conditional_middle else ''}]"
                f"  (cell = (stopped_count, {legacy_ylabel}))"
            ),
        )

    baseline_kwargs = dict(
        forced_curve=forced_curve,
        cal_tuned_forced_curve=cal_tuned_forced_curve,
        cal_tuned_forced_legend=cal_tuned_forced_legend,
        forced_fnr_cal_curve=forced_fnr_cal_curve,
        forced_fnr_cal_legend=forced_fnr_cal_legend,
        confidence_threshold_curve=confidence_threshold_curve,
        confidence_threshold_legend=confidence_threshold_legend,
        entropy_threshold_curve=entropy_threshold_curve,
        entropy_threshold_legend=entropy_threshold_legend,
    )

    if use_per_label:
        # 2×2: stopped | FNR_k1 / FNR_k2 | (BA if ≤2 labels)
        n_risk = len(ctrl_names)
        include_ba = n_risk <= 2
        n_panels = 1 + n_risk + (1 if include_ba else 0)
        nrows, ncols = 2, 2
        if n_panels > 4:
            ncols = int(np.ceil(n_panels / 2.0))
        fig, axes_grid = plt.subplots(
            nrows,
            ncols,
            figsize=COMPARE_FIGSIZE_2x2 if ncols == 2 else COMPARE_FIGSIZE_3COL,
            dpi=300,
            sharex=True,
            constrained_layout=False,
        )
        axes_flat = np.atleast_1d(axes_grid).ravel().tolist()
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        ax_stopped = axes_flat[0]
        _draw_stopped_fraction_ax(
            ax_stopped,
            sampling_rates,
            curves_by_method,
            casc_curve,
            n_test,
            **baseline_kwargs,
        )

        alpha_map = control_alpha_by_label or {}
        for i, name in enumerate(ctrl_names):
            ax = axes_flat[1 + i]
            flat = "_panel_metric"
            proj_methods = _project_method_curves_per_label(
                curves_by_method, nested_key, name, flat
            )
            proj_forced = _project_per_label_metric_curves(
                forced_curve, nested_key, name, flat
            )
            proj_cal = (
                _project_per_label_metric_curves(
                    cal_tuned_forced_curve, nested_key, name, flat
                )
                if cal_tuned_forced_curve is not None
                else None
            )
            proj_fnr_cal = (
                _project_per_label_metric_curves(
                    forced_fnr_cal_curve, nested_key, name, flat
                )
                if forced_fnr_cal_curve is not None
                else None
            )
            proj_conf = (
                _project_per_label_metric_curves(
                    confidence_threshold_curve, nested_key, name, flat
                )
                if confidence_threshold_curve is not None
                else None
            )
            proj_ent = (
                _project_per_label_metric_curves(
                    entropy_threshold_curve, nested_key, name, flat
                )
                if entropy_threshold_curve is not None
                else None
            )
            alpha_k = float(alpha_map.get(name, control_alpha))
            ylabel = f"Stopped {risk_word} ({label_display_name(name)})"
            _draw_controlled_risk_ax(
                ax,
                sampling_rates,
                proj_methods,
                casc_curve,  # unused for drawing; kept for signature compat
                proj_forced,
                flat,
                ylabel,
                control_alpha=alpha_k,
                risk_metric=risk_metric,
                cal_tuned_forced_curve=proj_cal,
                cal_tuned_forced_legend=cal_tuned_forced_legend,
                forced_fnr_cal_curve=proj_fnr_cal,
                forced_fnr_cal_legend=forced_fnr_cal_legend,
                confidence_threshold_curve=proj_conf,
                confidence_threshold_legend=confidence_threshold_legend,
                entropy_threshold_curve=proj_ent,
                entropy_threshold_legend=entropy_threshold_legend,
            )
            ax.set_ylabel(ylabel)

        axes_for_legend = axes_flat[:n_panels]
        if include_ba:
            ax_ba = axes_flat[1 + n_risk]
            ba_kwargs = {
                k: v for k, v in baseline_kwargs.items() if k != "forced_curve"
            }
            _draw_balanced_accuracy_ax(
                ax_ba,
                sampling_rates,
                curves_by_method,
                casc_curve,
                forced_curve,
                **ba_kwargs,
            )
            axes_for_legend = axes_flat[:n_panels]

        prio = _fastmri_legend_priority(
            cal_tuned_forced_legend,
            forced_fnr_cal_legend=forced_fnr_cal_legend,
            confidence_threshold_legend=confidence_threshold_legend,
            entropy_threshold_legend=entropy_threshold_legend,
        )
        # One α entry per distinct level (usually shared).
        alpha_vals = sorted({float(alpha_map.get(n, control_alpha)) for n in ctrl_names})
        for a in alpha_vals:
            prio.append(f"α = {a:g}")
        handles, labels = _merge_legend_by_priority(axes_for_legend, prio)
        ncol = len(handles)  # single horizontal row
        _top = FASTMRI_PANEL_LAYOUT_TOP
        fig.tight_layout(rect=[0.0, 0.0, 1.0, _top])
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, _top),
            ncol=ncol,
            fontsize=COMPARE_LEGEND_FONTSIZE * 0.82,
            borderaxespad=0.35,
            frameon=False,
            columnspacing=1.25,
            handletextpad=0.55,
            handlelength=2.0,
        )
        stem = f"compare_panel_{file_tag}"
        # Per-label panels are stopped-cohort metrics.
        stem += "_conditional"
        out_path = output_dir / f"{stem}.png"
        save_compare_figure(fig, out_path, dpi=300)
        return

    # Legacy 1×3 aggregate panel.
    fig, axes = plt.subplots(
        1, 3, figsize=COMPARE_FIGSIZE_3COL, dpi=300, sharex=True, constrained_layout=False
    )
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]

    _draw_stopped_fraction_ax(
        ax0,
        sampling_rates,
        curves_by_method,
        casc_curve,
        n_test,
        **baseline_kwargs,
    )
    _draw_controlled_risk_ax(
        ax1,
        sampling_rates,
        curves_by_method,
        casc_curve,
        forced_curve,
        legacy_key,
        legacy_ylabel,
        control_alpha=control_alpha,
        risk_metric=risk_metric,
        cal_tuned_forced_curve=cal_tuned_forced_curve,
        cal_tuned_forced_legend=cal_tuned_forced_legend,
        forced_fnr_cal_curve=forced_fnr_cal_curve,
        forced_fnr_cal_legend=forced_fnr_cal_legend,
        confidence_threshold_curve=confidence_threshold_curve,
        confidence_threshold_legend=confidence_threshold_legend,
        entropy_threshold_curve=entropy_threshold_curve,
        entropy_threshold_legend=entropy_threshold_legend,
    )
    _draw_balanced_accuracy_ax(
        ax2,
        sampling_rates,
        curves_by_method,
        casc_curve,
        forced_curve,
        cal_tuned_forced_curve=cal_tuned_forced_curve,
        cal_tuned_forced_legend=cal_tuned_forced_legend,
        forced_fnr_cal_curve=forced_fnr_cal_curve,
        forced_fnr_cal_legend=forced_fnr_cal_legend,
        confidence_threshold_curve=confidence_threshold_curve,
        confidence_threshold_legend=confidence_threshold_legend,
        entropy_threshold_curve=entropy_threshold_curve,
        entropy_threshold_legend=entropy_threshold_legend,
    )

    prio = _fastmri_legend_priority(
        cal_tuned_forced_legend,
        forced_fnr_cal_legend=forced_fnr_cal_legend,
        confidence_threshold_legend=confidence_threshold_legend,
        entropy_threshold_legend=entropy_threshold_legend,
    )
    prio.append(f"α = {float(control_alpha):g}")
    handles, labels = _merge_legend_by_priority([ax0, ax1, ax2], prio)
    ncol = len(handles)  # single horizontal row
    _top = 0.86
    fig.tight_layout(rect=[0.0, 0.0, 1.0, _top])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, _top),
        ncol=ncol,
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.92,
        borderaxespad=0.4,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.55,
        handlelength=2.1,
    )
    stem = f"compare_panel_{file_tag}"
    if plot_conditional_middle:
        stem += "_conditional"
    out_path = output_dir / f"{stem}.png"
    save_compare_figure(fig, out_path, dpi=300)


def plot_compare_metric(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    metric_key: str,
    metric_name: str,
    file_tag: str,
    *,
    control_alpha: Optional[float] = None,
    risk_metric: Optional[str] = None,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    forced_fnr_cal_curve: Optional[Dict[str, List[float]]] = None,
    forced_fnr_cal_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    """Generic per-method line plot for one of the FNR/FPR cumulative variants.

    When ``risk_metric`` matches the metric (FNR or FPR, conditional or unconditional),
    and ``control_alpha`` is set, draws a horizontal reference at α.
    """
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)

    aligned = False
    if control_alpha is not None and risk_metric is not None:
        if risk_metric == "fnr" and metric_key in (
            "fnr_cumulative",
            "fnr_unconditional_cumulative",
        ):
            aligned = True
        elif risk_metric == "fpr" and metric_key in (
            "fpr_cumulative",
            "fpr_unconditional_cumulative",
        ):
            aligned = True
    if aligned:
        draw_alpha_reference(ax, float(control_alpha))

    for key in COMPARE_CURVE_METHODS:
        ax.plot(
            sampling_rates,
            curves_by_method[key][metric_key],
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    ax.plot(
        sampling_rates,
        forced_curve[metric_key],
        marker=METHOD_MARKERS["forced"],
        linestyle=METHOD_LINESTYLES["forced"],
        color=METHOD_COLORS["forced"],
        label=METHOD_LABELS["forced"],
    )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        ax.plot(
            sampling_rates,
            cal_tuned_forced_curve[metric_key],
            marker=METHOD_MARKERS["cal_tuned_forced"],
            linestyle=METHOD_LINESTYLES["cal_tuned_forced"],
            color=METHOD_COLORS["cal_tuned_forced"],
            label=label,
        )
    if forced_fnr_cal_curve is not None:
        label_f = forced_fnr_cal_legend or METHOD_LABELS["forced_fnr_cal"]
        ax.plot(
            sampling_rates,
            forced_fnr_cal_curve[metric_key],
            marker=METHOD_MARKERS["forced_fnr_cal"],
            linestyle=METHOD_LINESTYLES["forced_fnr_cal"],
            color=METHOD_COLORS["forced_fnr_cal"],
            label=label_f,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        ax.plot(
            sampling_rates,
            confidence_threshold_curve[metric_key],
            marker=METHOD_MARKERS["confidence_threshold"],
            linestyle=METHOD_LINESTYLES["confidence_threshold"],
            color=METHOD_COLORS["confidence_threshold"],
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        ax.plot(
            sampling_rates,
            entropy_threshold_curve[metric_key],
            marker=METHOD_MARKERS["entropy_threshold"],
            linestyle=METHOD_LINESTYLES["entropy_threshold"],
            color=METHOD_COLORS["entropy_threshold"],
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel(metric_name)

    if metric_key in (
        "fnr_cumulative",
        "fnr_unconditional_cumulative",
        "fpr_cumulative",
        "fpr_unconditional_cumulative",
    ):
        ymin, ymax = _cumulative_risk_ylim(
            metric_key,
            curves_by_method,
            casc_curve,
            forced_curve,
            aligned=aligned,
            control_alpha=control_alpha,
            cal_tuned_forced_curve=cal_tuned_forced_curve,
            forced_fnr_cal_curve=forced_fnr_cal_curve,
            confidence_threshold_curve=confidence_threshold_curve,
            entropy_threshold_curve=entropy_threshold_curve,
        )
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=COMPARE_GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_{metric_key}_vs_sampling_rate_{file_tag}.png"
    save_compare_figure(fig, out_path)


def plot_compare_balanced_accuracy(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_curve: Dict[str, List[float]],
    forced_curve: Dict[str, List[float]],
    file_tag: str,
    *,
    cal_tuned_forced_curve: Optional[Dict[str, List[float]]] = None,
    cal_tuned_forced_legend: Optional[str] = None,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    """Macro balanced accuracy over labels on the cumulative stopped set (per method)."""
    metric_key = "balanced_accuracy_cumulative"
    metric_name = "Cumulative balanced accuracy"
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)
    for key in COMPARE_CURVE_METHODS:
        ax.plot(
            sampling_rates,
            curves_by_method[key][metric_key],
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    ax.plot(
        sampling_rates,
        forced_curve[metric_key],
        marker=METHOD_MARKERS["forced"],
        linestyle=METHOD_LINESTYLES["forced"],
        color=METHOD_COLORS["forced"],
        label=METHOD_LABELS["forced"],
    )
    if cal_tuned_forced_curve is not None:
        label = cal_tuned_forced_legend or METHOD_LABELS["cal_tuned_forced"]
        ax.plot(
            sampling_rates,
            cal_tuned_forced_curve[metric_key],
            marker=METHOD_MARKERS["cal_tuned_forced"],
            linestyle=METHOD_LINESTYLES["cal_tuned_forced"],
            color=METHOD_COLORS["cal_tuned_forced"],
            label=label,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        ax.plot(
            sampling_rates,
            confidence_threshold_curve[metric_key],
            marker=METHOD_MARKERS["confidence_threshold"],
            linestyle=METHOD_LINESTYLES["confidence_threshold"],
            color=METHOD_COLORS["confidence_threshold"],
            label=lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        ax.plot(
            sampling_rates,
            entropy_threshold_curve[metric_key],
            marker=METHOD_MARKERS["entropy_threshold"],
            linestyle=METHOD_LINESTYLES["entropy_threshold"],
            color=METHOD_COLORS["entropy_threshold"],
            label=lab_e,
        )
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=COMPARE_GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_balanced_accuracy_cumulative_vs_sampling_rate_{file_tag}.png"
    save_compare_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="FastMRI knee comparison: LTT-conditional (Holm) vs forced baselines."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/kspace/knee_vol_bayes_distill_freeze.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/gpfs/data/chopralab/us453/kspace_classifier/checkpoints/knee_vol_distill_freeze/s9xf9t2n/checkpoints/bayes_distill_arms-knee_vol-mrnet-epoch=6-best_metric=0.79.ckpt",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.075,
        help="Target risk level α (broadcast to all labels unless --alpha_per_label is set).",
    )
    parser.add_argument(
        "--alpha_per_label",
        type=str,
        default="0.075,0.075,0.1",
        help=(
            "Optional comma-separated per-label α_k (length K). Overrides broadcasting of --alpha "
            "for Option (i) LTT tests. Example: 0.1,0.15,0.2"
        ),
    )
    parser.add_argument("--delta", type=float, default=0.1, help="FWER level for FST/LTT.")
    parser.add_argument("--num_deltas", type=int, default=101)
    parser.add_argument("--est_size", type=int, default=50)
    parser.add_argument("--min_pos", type=int, default=2)
    parser.add_argument("--positive_class_idx", type=int, default=1)
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./fst_results/knee_vol_compare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling_rates", type=str, default="0.05,0.08,0.125,0.15,0.25")
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--num_lambdas", type=int, default=15)
    parser.add_argument("--use_est", action="store_true")
    parser.add_argument(
        "--cumulative_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--force_recompute_scores",
        action="store_true",
        help="Recompute and overwrite cached probabilities/labels.",
    )
    parser.add_argument(
        "--score_cache_base",
        type=str,
        default=None,
        help=(
            "If set, read/write score caches under <dir>/score_cache/seed_<seed>/ instead of under "
            "the resolved output directory (after testset/). Same layout as the default cache tree."
        ),
    )
    parser.add_argument(
        "--risk_metric",
        type=str,
        default="fnr",
        choices=["fnr", "fpr"],
        help="Risk controlled by LTT: class-conditional FNR (default) or FPR, per label.",
    )
    parser.add_argument(
        "--control_fnr_and_fpr",
        action="store_true",
        help="Take p(λ)=max over all K FNR and K FPR class-conditional p-values (2K tests).",
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        default=None,
        help=(
            "Comma-separated label names to omit from Option (i) FNR/FPR control "
            "(still used for stopping). Example: acl. Mutually exclusive with --control_labels."
        ),
    )
    parser.add_argument(
        "--control_labels",
        type=str,
        default=None,
        help=(
            "Comma-separated label names to include in Option (i) control "
            "(default: all). Example: mtear,abnormal. Mutually exclusive with --exclude_labels."
        ),
    )
    parser.add_argument(
        "--abstain_penalty",
        type=float,
        default=0.03,
        help="Loss assigned to non-stopped (abstained) calibration samples for LTT-abstain.",
    )
    parser.add_argument(
        "--ltt_abstain_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
        help="Utility for LTT with abstention cost (default: stop as many as possible).",
    )
    parser.add_argument(
        "--ltt_conditional_utility",
        type=str,
        default="balanced_accuracy_stopped",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
        help="Utility on LTT safe set: stopped_count | neg_mean_stop_index | neg_calib_risk | balanced_accuracy_stopped.",
    )
    parser.add_argument("--default_lambda_idx_abstain", type=int, default=None)
    parser.add_argument("--default_lambda_idx_conditional", type=int, default=None)
    parser.add_argument(
        "--ltt_conditional_test",
        type=str,
        default="ratio",
        choices=["ratio", "via_W"],
        help="Deprecated (ignored). Option (i) always uses class-conditional binomial p-values.",
    )
    parser.add_argument(
        "--ltt_pvalue_method",
        type=str,
        default="binomial",
        choices=["binomial", "hoeffding"],
        help="Per-label p-value for Option (i): exact Binomial CDF (default) or Hoeffding.",
    )
    parser.add_argument(
        "--no_cal_tuned_forced",
        action="store_true",
        help="Omit forced-with-calibrated-thresholds baseline (otherwise on by default).",
    )
    parser.add_argument(
        "--cal_tuned_forced_metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy"],
        help=(
            "When fitting cal-tuned forced: each label's threshold maximizes this on calibration "
            "at that sampling step (default: accuracy)."
        ),
    )
    parser.add_argument(
        "--cal_tuned_forced_grid",
        type=int,
        default=101,
        help="Uniformly spaced θ candidates in [0,1] for cal-tuned threshold search.",
    )
    parser.add_argument(
        "--forced_fnr_calibrated",
        action="store_true",
        help=(
            "Add forced baseline with per-(t,k) θ from calibration: largest grid θ such that "
            "empirical FNR on calibration positives for that label is at most the cap "
            "(see --forced_fnr_cal_alpha, --forced_fnr_match_casc_schedule)."
        ),
    )
    parser.add_argument(
        "--forced_fnr_cal_alpha",
        type=float,
        default=None,
        help="FNR cap for that calibration search (defaults to --alpha). Only with --forced_fnr_calibrated.",
    )
    parser.add_argument(
        "--forced_fnr_match_casc_schedule",
        action="store_true",
        help=(
            "Use alpha/T per timestep (same uniform split as Bonferroni dual-threshold calibration) "
            "instead of the same cap at every sampling step."
        ),
    )
    parser.add_argument(
        "--forced_fnr_cal_grid",
        type=int,
        default=101,
        help="Uniformly spaced θ grid for FN-capped forced threshold search.",
    )
    parser.add_argument(
        "--plot_conditional",
        action="store_true",
        help=(
            "Middle panel subplot: cumulative FNR/FPR on cumulative stopped cohort only "
            "(not zero-filled unconditional). Saves compare_panel_<tag>_conditional.png."
        ),
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help=(
            "Heuristic baseline: each sample stops at the first sampling rate where every label's "
            "positive score is ≥ this threshold. Pass a negative value to disable."
        ),
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.2,
        help=(
            "Heuristic baseline: stop at the first rate where mean per-label binary entropy "
            "(from p and 1−p) is ≤ this value. Pass a negative value to disable."
        ),
    )
    parser.add_argument(
        "--print_ambiguity_bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print b_neg/b_pos from compute_bands per sampling-rate and label "
            "(before λ via compute_bounds). Use --no-print_ambiguity_bands to silence."
        ),
    )
    add_ambiguity_band_args(parser)
    args = parser.parse_args()
    configure_compare_plot_style()

    # Seed all RNGs before any model is constructed so that random sub-modules
    # whose weights are *not* restored by load_from_checkpoint(strict=False)
    # get a reproducible initialization.
    import random as _py_random
    _py_random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    config = get_config(args.config_path)
    device = torch.device(args.device)
    label_names = config.label_names.split(",") if isinstance(config.label_names, str) else config.label_names
    input_type = getattr(config, "input_type", "kspace")
    is_volume = "vol" in str(getattr(config, "dataset", "")).lower()
    dataset_tag = "knee_vol" if is_volume else "knee_slice"

    val_ds, test_ds = get_val_test_datasets(config, args.use_testset)
    batch_size = args.batch_size or getattr(config, "val_batch_size", getattr(config, "batch_size", 32))

    label_map = collect_labels(val_ds, label_names, batch_size, args.num_workers)
    label_matrix_val = build_label_matrix(label_map, label_names, args.positive_class_idx)
    if args.use_est:
        est_ds, remaining_val = split_estimation_set(
            val_ds, label_matrix_val, args.est_size, args.seed, args.min_pos
        )
    else:
        est_ds = None
        remaining_val = val_ds

    if args.use_testset and test_ds is not None:
        calib_ds = remaining_val
    else:
        calib_ds, test_ds = split_dataset(remaining_val, args.calib_fraction, args.seed)
    if not args.use_est:
        est_ds = calib_ds

    model = load_model(config, args.checkpoint_path, device)
    sampling_rates = [float(x) for x in args.sampling_rates.split(",") if x.strip()]
    lambda_candidates = np.linspace(0.0, 1.0, args.num_lambdas)

    out_dir = Path(args.output_dir)
    if args.use_testset:
        out_dir = out_dir / "testset"
    out_dir.mkdir(parents=True, exist_ok=True)

    est_loader = build_dataloader(est_ds, batch_size, args.num_workers, shuffle=False)
    calib_loader = build_dataloader(calib_ds, batch_size, args.num_workers, shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size, args.num_workers, shuffle=False)

    est_mask_state = CumulativeMaskState(len(est_ds)) if args.cumulative_sampling else None
    calib_mask_state = CumulativeMaskState(len(calib_ds)) if args.cumulative_sampling else None
    test_mask_state = CumulativeMaskState(len(test_ds)) if args.cumulative_sampling else None
    score_cache_root = Path(args.score_cache_base).expanduser() if args.score_cache_base else out_dir
    score_cache_dir = build_score_cache_dir(str(score_cache_root), args.seed)

    est_pos_scores, calib_pos_scores, test_pos_scores = [], [], []
    label_matrix_est = None
    label_matrix_cal = None
    label_matrix_test = None
    for sampling_rate in sampling_rates:
        mask_fn = build_mask_for_sampling_rate(config, sampling_rate)
        cache_path = score_cache_dir / f"sr_{sampling_rate:.6f}.pt"
        cached = None if args.force_recompute_scores else load_cached_scores(cache_path)
        cache_ok = False
        if cached is not None:
            cache_ok = (
                cached.get("sampling_rate") == sampling_rate
                and cached.get("seed") == args.seed
                and cached.get("calib_fraction") == args.calib_fraction
                and cached.get("use_testset") == args.use_testset
                and cached.get("label_names") == label_names
                and cached.get("positive_class_idx") == args.positive_class_idx
                and cached.get("use_est") == args.use_est
                and cached.get("est_size") == len(est_ds)
                and cached.get("min_pos") == args.min_pos
            )
        if cache_ok:
            est_probs = cached["est_probs"]
            est_labels = cached["est_labels"]
            cal_probs = cached["cal_probs"]
            cal_labels = cached["cal_labels"]
            tst_probs = cached["test_probs"]
            tst_labels = cached["test_labels"]
        else:
            est_probs, est_labels = get_probs_labels(
                model, est_loader, label_names, device, mask_fn,
                cumulative_state=est_mask_state, input_type=input_type, is_volume=is_volume,
            )
            cal_probs, cal_labels = get_probs_labels(
                model, calib_loader, label_names, device, mask_fn,
                cumulative_state=calib_mask_state, input_type=input_type, is_volume=is_volume,
            )
            tst_probs, tst_labels = get_probs_labels(
                model, test_loader, label_names, device, mask_fn,
                cumulative_state=test_mask_state, input_type=input_type, is_volume=is_volume,
            )
            save_cached_scores(
                cache_path,
                {
                    "sampling_rate": sampling_rate,
                    "seed": args.seed,
                    "calib_fraction": args.calib_fraction,
                    "use_testset": args.use_testset,
                    "label_names": label_names,
                    "positive_class_idx": args.positive_class_idx,
                    "use_est": args.use_est,
                    "est_size": len(est_ds),
                    "min_pos": args.min_pos,
                    "est_probs": est_probs,
                    "est_labels": est_labels,
                    "cal_probs": cal_probs,
                    "cal_labels": cal_labels,
                    "test_probs": tst_probs,
                    "test_labels": tst_labels,
                },
            )
        est_pos_scores.append(build_prob_matrix(est_probs, label_names, args.positive_class_idx))
        calib_pos_scores.append(build_prob_matrix(cal_probs, label_names, args.positive_class_idx))
        test_pos_scores.append(build_prob_matrix(tst_probs, label_names, args.positive_class_idx))
        if label_matrix_est is None:
            label_matrix_est = build_label_matrix(est_labels, label_names, args.positive_class_idx)
            label_matrix_cal = build_label_matrix(cal_labels, label_names, args.positive_class_idx)
            label_matrix_test = build_label_matrix(tst_labels, label_names, args.positive_class_idx)

    est_pos_scores = np.stack(est_pos_scores, axis=1)
    calib_pos_scores = np.stack(calib_pos_scores, axis=1)
    test_pos_scores = np.stack(test_pos_scores, axis=1)

    b_pos, b_neg = compute_bands_from_argparse(est_pos_scores, label_matrix_est, args)
    if args.print_ambiguity_bands:
        print_ambiguity_band_diagnostics(
            sampling_rates,
            b_pos,
            b_neg,
            label_names,
            args.num_deltas,
            band_source=str(args.band_source),
        )
    losses_mat, stopped_mat, stop_idx_mat, preds_cube = per_lambda_calibration_arrays_knee_metric(
        calib_pos_scores, label_matrix_cal, b_pos, b_neg, lambda_candidates, args.risk_metric
    )

    alpha_per_label = parse_alpha_per_label(args.alpha_per_label)
    alpha_k = resolve_alpha_per_label(
        float(args.alpha), alpha_per_label, int(label_matrix_cal.shape[1])
    )
    control_idxs = resolve_control_label_indices(
        list(label_names),
        control_labels=args.control_labels,
        exclude_labels=args.exclude_labels,
    )
    print(
        f"Option (i) α_k ({len(alpha_k)} labels): "
        + ", ".join(f"{n}={float(a):g}" for n, a in zip(label_names, alpha_k)),
        flush=True,
    )
    print(
        "Option (i) controlled labels: "
        + ", ".join(label_names[i] for i in control_idxs),
        flush=True,
    )

    method_selections = select_lambda_methods(
        lambda_candidates,
        losses_mat,
        stopped_mat,
        stop_idx_mat,
        args.alpha,
        args.delta,
        args.abstain_penalty,
        args.ltt_abstain_utility,
        args.ltt_conditional_utility,
        args.default_lambda_idx_abstain,
        args.default_lambda_idx_conditional,
        args.ltt_conditional_test,
        preds_cube=preds_cube,
        labels=label_matrix_cal,
        risk_metric=str(args.risk_metric),
        alpha_per_label=alpha_k,
        pvalue_method=str(args.ltt_pvalue_method),
        control_both=bool(args.control_fnr_and_fpr),
        control_label_indices=control_idxs,
    )

    m_grid = int(lambda_candidates.shape[0])
    holm_tightest = float(args.delta) / max(m_grid, 1)
    print(
        f"dataset={dataset_tag} risk_metric={args.risk_metric} "
        f"LTT=class-conditional-{args.ltt_pvalue_method} "
        f"Grid m={m_grid}, delta={args.delta}. "
        f"LTT-Cond uses Holm; tightest per-step cutoff = delta/m = {holm_tightest:.3g}."
    )
    cond_fingerprint = hash(
        (
            args.alpha,
            args.delta,
            args.risk_metric,
            args.ltt_pvalue_method,
            args.control_fnr_and_fpr,
            tuple(float(x) for x in alpha_k.tolist()),
            args.num_lambdas,
            args.seed,
            args.est_size,
            args.min_pos,
            args.calib_fraction,
            args.ltt_conditional_utility,
            args.default_lambda_idx_conditional,
            int(np.asarray(preds_cube).tobytes().__hash__()),
            int(np.asarray(stopped_mat).tobytes().__hash__()),
        )
    )
    print(f"LTT-conditional input fingerprint: {cond_fingerprint}")
    for key, sel in method_selections.items():
        lam = sel["lambda_hat"]
        lam_str = "None" if lam is None else f"{lam:.6f}"
        risks_arr = np.asarray(sel["risks"], dtype=float)
        ns_arr = np.asarray(sel["ns"], dtype=int)
        p_arr = np.asarray(sel["p_values"], dtype=float)
        if risks_arr.size == 0:
            print(f"{METHOD_LABELS[key]}: empty risk grid")
            continue
        j_min_r = int(np.argmin(risks_arr))
        j_min_p = int(np.argmin(p_arr))
        print(
            f"{METHOD_LABELS[key]}: lambda_hat={lam_str} reason={sel['meta']['reason']}"
        )
        print(
            f"  min max_k R_k={risks_arr[j_min_r]:.4f} at lambda={lambda_candidates[j_min_r]:.4f} "
            f"(min_k n_k={int(ns_arr[j_min_r])}, p=max_k p_k={p_arr[j_min_r]:.3g})"
        )
        print(
            f"  min p_value={p_arr[j_min_p]:.3g} at lambda={lambda_candidates[j_min_p]:.4f} "
            f"(max_k R_k={risks_arr[j_min_p]:.4f}, min_k n_k={int(ns_arr[j_min_p])})"
        )
        rejected = np.asarray(sel["meta"].get("rejected", []), dtype=bool)
        print(
            f"  Holm rejected {int(rejected.sum())} / {m_grid} "
            f"(tightest cutoff delta/m={holm_tightest:.3g})"
        )
        details = sel["meta"].get("class_conditional_details")
        if details is not None:
            print_class_conditional_label_diagnostics(
                lambda_candidates,
                details,
                list(label_names),
                delta=float(args.delta),
            )

    curves_by_method: Dict[str, Dict[str, List[float]]] = {}
    for key in LAMBDA_METHODS:
        curves_by_method[key] = lambda_method_curves(
            method_selections[key]["lambda_hat"],
            test_pos_scores,
            label_matrix_test,
            b_pos,
            b_neg,
            label_indices=control_idxs,
            label_names=label_names,
        )
    # Nominal band baseline omitted from plots / summary for now.
    # curves_by_method["nominal_bands"] = lambda_method_curves(
    #     1.0,
    #     test_pos_scores,
    #     label_matrix_test,
    #     b_pos,
    #     b_neg,
    # )

    casc_curve = casc_curves(
        calib_pos_scores,
        label_matrix_cal,
        test_pos_scores,
        label_matrix_test,
        args.alpha,
        args.positive_class_idx,
        label_indices=control_idxs,
        label_names=label_names,
    )
    forced_curve = forced_curves(
        test_pos_scores,
        label_matrix_test,
        label_indices=control_idxs,
        label_names=label_names,
    )

    cal_tuned_curve: Optional[Dict[str, List[float]]] = None
    cal_tuned_thresholds: Optional[np.ndarray] = None
    cal_tuned_legend: Optional[str] = None
    if not bool(args.no_cal_tuned_forced):
        cal_tuned_curve, cal_tuned_thresholds = cal_tuned_forced_curves(
            calib_pos_scores,
            label_matrix_cal,
            test_pos_scores,
            label_matrix_test,
            tuning_metric=str(args.cal_tuned_forced_metric),
            threshold_grid_pts=int(args.cal_tuned_forced_grid),
            label_indices=control_idxs,
            label_names=label_names,
        )
        cal_tuned_legend = cal_tuned_forced_legend_label(str(args.cal_tuned_forced_metric))

    forced_fnr_curve: Optional[Dict[str, List[float]]] = None
    forced_fnr_thresholds: Optional[np.ndarray] = None
    forced_fnr_legend: Optional[str] = None
    if bool(args.forced_fnr_calibrated):
        t_steps_mx = int(test_pos_scores.shape[1])
        alpha_fnr_cap = (
            float(args.forced_fnr_cal_alpha)
            if args.forced_fnr_cal_alpha is not None
            else float(args.alpha)
        )
        if bool(args.forced_fnr_match_casc_schedule):
            fnr_caps_sched = build_uniform_alpha_schedule(t_steps_mx, alpha_fnr_cap)
        else:
            fnr_caps_sched = np.full(t_steps_mx, alpha_fnr_cap, dtype=float)
        forced_fnr_curve, forced_fnr_thresholds = fnr_cap_constrained_forced_cal_curves(
            calib_pos_scores,
            label_matrix_cal,
            test_pos_scores,
            label_matrix_test,
            fnr_cap_per_timestep=fnr_caps_sched,
            threshold_grid_pts=int(args.forced_fnr_cal_grid),
            label_indices=control_idxs,
            label_names=label_names,
        )
        forced_fnr_legend = forced_fnr_cal_legend_label(
            alpha_ref=alpha_fnr_cap,
            match_casc_schedule=bool(args.forced_fnr_match_casc_schedule),
            num_timesteps=t_steps_mx,
        )

    conf_thr = (
        float(args.confidence_threshold)
        if args.confidence_threshold is not None and float(args.confidence_threshold) >= 0.0
        else None
    )
    ent_thr = (
        float(args.entropy_threshold)
        if args.entropy_threshold is not None and float(args.entropy_threshold) >= 0.0
        else None
    )
    conf_thr_curve, ent_thr_curve = fixed_threshold_schedule_curves(
        test_pos_scores,
        label_matrix_test,
        confidence_threshold=conf_thr,
        entropy_threshold=ent_thr,
        label_indices=control_idxs,
        label_names=list(label_names),
    )
    conf_thr_legend = (
        confidence_baseline_legend_label(float(conf_thr))
        if conf_thr_curve is not None and conf_thr is not None
        else None
    )
    ent_thr_legend = (
        entropy_baseline_legend_label(float(ent_thr))
        if ent_thr_curve is not None and ent_thr is not None
        else None
    )

    file_tag = f"{dataset_tag}_{args.risk_metric}"
    n_test = int(label_matrix_test.shape[0])
    control_label_names = [label_names[i] for i in control_idxs]
    control_alpha_by_label = {
        label_names[i]: float(alpha_k[i]) for i in control_idxs
    }
    label_prevalence_rows = [
        (
            str(name),
            int(np.sum(label_matrix_test[:, k] == 1)),
            int(label_matrix_test.shape[0]),
        )
        for k, name in enumerate(label_names)
    ]
    plot_compare_fastmri_panel(
        out_dir,
        sampling_rates,
        curves_by_method,
        casc_curve,
        forced_curve,
        file_tag,
        n_test,
        risk_metric=str(args.risk_metric),
        control_alpha=float(np.max(alpha_k)),
        cal_tuned_forced_curve=cal_tuned_curve,
        cal_tuned_forced_legend=cal_tuned_legend,
        forced_fnr_cal_curve=forced_fnr_curve,
        forced_fnr_cal_legend=forced_fnr_legend,
        confidence_threshold_curve=conf_thr_curve,
        confidence_threshold_legend=conf_thr_legend,
        entropy_threshold_curve=ent_thr_curve,
        entropy_threshold_legend=ent_thr_legend,
        plot_conditional_middle=bool(args.plot_conditional),
        control_label_names=control_label_names,
        control_alpha_by_label=control_alpha_by_label,
        label_prevalence_rows=label_prevalence_rows,
        label_prevalence_split="test",
    )

    out_summary = {
        "dataset": dataset_tag,
        "plot_conditional": bool(args.plot_conditional),
        "alpha": args.alpha,
        "alpha_per_label": alpha_k.tolist(),
        "delta": args.delta,
        "n_cal": int(label_matrix_cal.shape[0]),
        "n_est": int(label_matrix_est.shape[0]),
        "n_test": int(label_matrix_test.shape[0]),
        "risk_metric": args.risk_metric,
        "abstain_penalty": float(args.abstain_penalty),
        "ltt_abstain_utility": args.ltt_abstain_utility,
        "ltt_conditional_utility": args.ltt_conditional_utility,
        "ltt_conditional_test": args.ltt_conditional_test,
        "ltt_pvalue_method": args.ltt_pvalue_method,
        "control_fnr_and_fpr": bool(args.control_fnr_and_fpr),
        "exclude_labels": args.exclude_labels,
        "control_labels": args.control_labels,
        "control_label_indices": control_idxs,
        "control_label_names": [label_names[i] for i in control_idxs],
        "confidence_threshold": args.confidence_threshold,
        "entropy_threshold": args.entropy_threshold,
        "score_cache_base": args.score_cache_base,
        "score_cache_dir": str(score_cache_dir),
        "default_lambda_idx_abstain": args.default_lambda_idx_abstain,
        "default_lambda_idx_conditional": args.default_lambda_idx_conditional,
        "ambiguity_band": triage_band_config_dict(args),
        "selections": {
            **{
                key: (
                    {
                        "lambda_hat": (
                            None if sel["lambda_hat"] is None else float(sel["lambda_hat"])
                        ),
                        "reason": sel["meta"]["reason"],
                    }
                    | (
                        {"conditional_test": sel["conditional_test"]}
                        if key == "ltt_conditional"
                        else {}
                    )
                )
                for key, sel in method_selections.items()
            },
            # "nominal_bands": {
            #     "lambda_hat": 1.0,
            #     "reason": "fixed_nominal_overlap_from_estimation",
            # },
        },
        "curves": {
            "sampling_rates": sampling_rates,
            **curves_by_method,
            "casc": casc_curve,
            "forced": forced_curve,
            **(
                {
                    "cal_tuned_forced": cal_tuned_curve,
                    "cal_tuned_forced_thresholds": cal_tuned_thresholds.tolist(),
                }
                if cal_tuned_curve is not None and cal_tuned_thresholds is not None
                else {}
            ),
            **(
                {
                    "forced_fnr_calibrated": forced_fnr_curve,
                    "forced_fnr_calibrated_thresholds": forced_fnr_thresholds.tolist(),
                    "forced_fnr_calibrated_meta": {
                        "alpha_cap": (
                            float(args.forced_fnr_cal_alpha)
                            if args.forced_fnr_cal_alpha is not None
                            else float(args.alpha)
                        ),
                        "match_casc_schedule": bool(args.forced_fnr_match_casc_schedule),
                        "grid_pts": int(args.forced_fnr_cal_grid),
                    },
                }
                if forced_fnr_curve is not None and forced_fnr_thresholds is not None
                else {}
            ),
            **(
                {"confidence_threshold_baseline": conf_thr_curve}
                if conf_thr_curve is not None
                else {}
            ),
            **(
                {"entropy_threshold_baseline": ent_thr_curve}
                if ent_thr_curve is not None
                else {}
            ),
        },
    }

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    out_path = out_dir / f"compare_knee_summary_{file_tag}.json"
    with open(out_path, "w") as handle:
        json.dump(out_summary, handle, indent=2, default=_json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
