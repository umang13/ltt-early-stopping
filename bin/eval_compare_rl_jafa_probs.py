#!/usr/bin/env python3
"""
CIFAR-10 (JAFA / RL): Compare raw RL stopping vs post hoc wrappers on RL-exported trajectories.

Loads NPZ splits (RL-feature-selection/scripts/export_cifar_score_matrices.py).

**Wrappers** (LTT-conditional; selectable λ_hat):
  - Default `--wrapper_signal probs`: identical to eval_compare_cifar10 on `probs[:,t,:]`
    (`max_score` or `set_size` via `--stopping_rule`).
  - `--wrapper_signal q_margin_max` | `q_margin_mean`: first time at which the exported
    `score_q_margin_*` satisfies `score >= λ * baseline_t`, with `baseline_t` the per-column
    (1 − delta_est)-quantile of finite scores on the estimation subset (Nan entries never stop).

Calibration risk on stopped samples remains `loss_kind`; Q-margin modes require
`loss_kind argmax_miscoverage` (prediction = argmax of `probs` at stop time).

**Bonferroni:** ``casc_curve`` is still computed for the JSON summary only (no Bonferroni line on figures).

**JAFA baseline**: RL STOP via `natural_stop_t`.

Artifacts are written under ``<output_dir>/max-prob/``, ``q-margin-max/``, or ``q-margin-mean/`` depending on ``--wrapper_signal``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_figure_export import (
    style_compare_ax,
    COMPARE_FIGSIZE_2COL,
    COMPARE_FIGSIZE_SINGLE,
    COMPARE_LEGEND_FONTSIZE,
    COMPARE_RCS_LEGEND_LABEL,
    COMPARE_XLABEL_SAMPLING,
    configure_compare_plot_style,
    draw_accuracy_target_reference,
    plot_compare_series,
    print_compare_curve_table,
    save_compare_figure,
)

from eval_compare_cifar10 import select_lambda_methods
from eval_fst_cifar10 import (
    argmax_miscoverage_losses_from_stops,
    build_max_score_baseline,
    build_true_class_baseline,
    casc_curve,
    forced_curve,
    fst_curve,
    per_lambda_calibration_arrays,
)

WRAPPER_SIGNAL_NPZ_KEYS = {
    "probs": None,
    "q_margin_max": "score_q_margin_max",
    "q_margin_mean": "score_q_margin_mean",
}

# Saved under args.output_dir / <subdir>/ so parallel runs separate without clobbering.
WRAPPER_SIGNAL_OUTPUT_SUBDIR = {
    "probs": "max-prob",
    "q_margin_max": "q-margin-max",
    "q_margin_mean": "q-margin-mean",
}

METHOD_COLORS = {
    "jafa": "#DD8452",
    "ltt_conditional": "#8172B2",
    "casc": "#55A868",
    "forced": "#7F7F7F",
}

METHOD_LABELS = {
    "jafa": "JAFA",
    "ltt_conditional": f"JAFA + {COMPARE_RCS_LEGEND_LABEL}",
    "casc": "JAFA + Bonferroni",
    "forced": "Forced",
}

METHOD_LINESTYLES = {
    "jafa": "-",
    "ltt_conditional": "--",
    "casc": "-",
    "forced": "-",
}

METHOD_MARKERS = {
    "jafa": "s",
    "ltt_conditional": "D",
    "casc": "o",
    "forced": "o",
}

# Only these two appear on the main comparison panel (plus optional reference 1−α line).
ORDER_KEYS_RL = ["jafa", "ltt_conditional"]


def _merge_legend_rl(axes: List[Any], label_priority: List[str]) -> Tuple[List, List]:
    by_label: Dict[str, object] = {}
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for h, lab in zip(hs, ls):
            if lab not in by_label:
                by_label[lab] = h
    ordered_h: List = []
    ordered_l: List[str] = []
    for lab in label_priority:
        if lab in by_label:
            ordered_h.append(by_label[lab])
            ordered_l.append(lab)
    for lab, h in by_label.items():
        if lab not in ordered_l:
            ordered_h.append(h)
            ordered_l.append(lab)
    return ordered_h, ordered_l


def plot_rl_compare_panel(
    output_dir: Path,
    acquisition_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    accuracy_by_method: Dict[str, Dict[str, List[float]]],
    plot_tag: str,
    n_test: int,
    alpha_control: float,
    loss_kind: str,
    plot_conditional: bool = False,
) -> None:
    """One row, two subplots; JAFA and JAFA+LTT-Cond only, plus reference 1−α when applicable."""
    table_rows = [
        (METHOD_LABELS[key], curves_by_method[key]) for key in ORDER_KEYS_RL
    ]
    print_compare_curve_table(
        acquisition_rates,
        table_rows,
        metric_key="accuracy",
        metric_label="stopped-sample accuracy",
        title=(
            f"Plot data table [{plot_tag}"
            f"{'_conditional' if plot_conditional else ''}]"
            "  (cell = (stopped_count, stopped-sample accuracy))"
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=COMPARE_FIGSIZE_2COL, dpi=300, sharex=True, constrained_layout=False)
    ax0, ax1 = axes[0], axes[1]
    inv = 1.0 / float(n_test)

    def _series(ax, key: str, ys, lab: str, zorder: int = 2) -> None:
        plot_compare_series(
            ax,
            acquisition_rates,
            ys,
            color=METHOD_COLORS[key],
            linestyle=METHOD_LINESTYLES[key],
            marker=METHOD_MARKERS[key],
            label=lab,
            zorder=zorder,
        )

    for key in ORDER_KEYS_RL:
        _series(
            ax0,
            key,
            [int(c) * inv for c in curves_by_method[key]["stopped_count"]],
            METHOD_LABELS[key],
            zorder=3 if key == "ltt_conditional" else 2,
        )
    ax0.set_ylabel("Fraction stopped")
    ax0.set_ylim(0.0, 1.05)
    style_compare_ax(ax0, grid=True)
    ax0.set_xlabel(COMPARE_XLABEL_SAMPLING)

    target_acc = float(max(0.0, min(1.0, 1.0 - alpha_control)))
    prio = [METHOD_LABELS[k] for k in ORDER_KEYS_RL]
    if loss_kind == "argmax_miscoverage":
        ref_lab = f"1−α = {target_acc:g}"
        draw_accuracy_target_reference(ax1, target_acc, label=ref_lab)
        prio = [ref_lab] + prio

    for key in ORDER_KEYS_RL:
        _series(
            ax1,
            key,
            accuracy_by_method[key]["accuracy"],
            METHOD_LABELS[key],
            zorder=3 if key == "ltt_conditional" else 2,
        )

    ax1.set_ylabel("Accuracy of stopped samples")
    style_compare_ax(ax1, grid=True)
    ax1.set_xlabel(COMPARE_XLABEL_SAMPLING)

    handles, labels = _merge_legend_rl([ax0, ax1], prio)
    _top = 0.82
    fig.tight_layout(rect=[0.0, 0.0, 1.0, _top])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, _top),
        ncol=len(handles),
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.92,
        borderaxespad=0.35,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.55,
        handlelength=2.1,
    )
    tag_out = plot_tag
    if plot_conditional:
        tag_out += "_conditional"
    out_path = output_dir / f"compare_panel_{tag_out}.png"
    save_compare_figure(fig, out_path, dpi=300)


def load_scores_npz(score_path: Path) -> Dict[str, np.ndarray]:
    with np.load(score_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path) as handle:
        return json.load(handle)


def column_indices_for_visible_counts(
    visible_counts: np.ndarray, wanted_counts: List[int]
) -> np.ndarray:
    lookup = {int(c): idx for idx, c in enumerate(np.asarray(visible_counts).ravel().tolist())}
    missing = [c for c in wanted_counts if int(c) not in lookup]
    if missing:
        raise ValueError(f"wanted_counts not in visible_counts: {missing[:10]}...")
    return np.asarray([lookup[int(c)] for c in wanted_counts], dtype=np.int64)


def uniform_subsample_column_indices(n_columns: int, n_keep: int) -> np.ndarray:
    """~evenly spaced column indices along the trajectory (reduces Casc / Bonferroni depth)."""
    nc = int(n_columns)
    nk = int(n_keep)
    if nk < 2:
        raise ValueError("--num_sampling_rates must be >= 2 when set")
    if nc <= 1:
        return np.array([0], dtype=np.int64)
    if nk >= nc:
        return np.arange(nc, dtype=np.int64)
    pos = np.linspace(0.0, float(nc - 1), num=nk)
    idx = np.unique(np.clip(np.round(pos).astype(np.int64), 0, nc - 1))
    return idx.astype(np.int64, copy=False)


def split_val_into_est_calib(
    n_val: int,
    est_size: int,
    calib_fraction: float,
    seed: int,
    use_full_remaining_as_calib: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirrors eval_fst_cifar10.split_estimation_set + split_dataset ordering on row indices.

    If use_full_remaining_as_calib=True (default when test NPZ is separate), all indices after
    the estimation draw are used for calibration.
    """
    if n_val <= 0:
        raise ValueError("n_val must be positive")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_val)
    est_n = min(int(est_size), n_val)
    est_idx = perm[:est_n]
    rem = perm[est_n:]
    if rem.size == 0:
        raise ValueError("No samples left for calibration after estimation split")
    if use_full_remaining_as_calib:
        calib_idx = rem.astype(np.int64, copy=False)
        return est_idx.astype(np.int64, copy=False), calib_idx

    rng2 = np.random.default_rng(int(seed))
    order = rng2.permutation(rem.size)
    rem_ord = rem[order]
    frac = float(calib_fraction)
    n_calib = int(np.floor(rem_ord.size * frac))
    if n_calib <= 0:
        raise ValueError(
            f"calibration split empty (n_rem={rem_ord.size}, calib_fraction={frac})"
        )
    calib_idx = rem_ord[:n_calib].astype(np.int64, copy=False)
    return est_idx.astype(np.int64, copy=False), calib_idx


def slice_trajectory_bundle(
    arrays: Dict[str, np.ndarray],
    col_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key == "visible_counts":
            out[key] = np.asarray(value).ravel()[col_idx]
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2 and arr.shape[1] == len(np.asarray(arrays["visible_counts"]).ravel()):
            out[key] = arr[:, col_idx]
        else:
            out[key] = arr
    return out


def build_scalar_margin_baseline(
    est_scores: np.ndarray,
    delta_est: float,
) -> np.ndarray:
    """
    High quantiles of JAFA/Q-type margins per timestep (finite values only).
    Matches the role of build_max_score_baseline but for arbitrary scalar curves.
    """
    est_scores = np.asarray(est_scores, dtype=np.float64)
    n_steps = est_scores.shape[1]
    out = np.zeros(n_steps, dtype=np.float64)
    q_level = float(max(0.0, min(1.0, 1.0 - float(delta_est))))
    for t in range(n_steps):
        col = est_scores[:, t]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[t] = np.nan
            continue
        out[t] = float(np.quantile(finite, q_level))
    return out


def stop_by_scalar_margin_ge(
    scores: np.ndarray,
    thresholds_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Same temporal structure as stop_by_max_score: first finite t with scores[:,t] >= thresholds_t[t]."""
    scores = np.asarray(scores, dtype=np.float64)
    n, t_steps = scores.shape
    thresholds_t = np.asarray(thresholds_t, dtype=np.float64)
    stop_indices = np.full(n, t_steps - 1, dtype=int)
    stopped = np.zeros(n, dtype=bool)
    for t in range(t_steps):
        thr = thresholds_t[t]
        s_col = scores[:, t]
        valid = np.isfinite(s_col) & np.isfinite(thr)
        stop_mask = valid & (s_col >= thr)
        newly = (~stopped) & stop_mask
        stop_indices[newly] = t
        stopped[newly] = True
    return stop_indices, stopped


def per_lambda_calibration_scalar_margin(
    probs: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    lambdas: np.ndarray,
    baseline_t: np.ndarray,
    loss_kind: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if loss_kind != "argmax_miscoverage":
        raise ValueError(
            "wrapper_signal q_margin_* requires loss_kind argmax_miscoverage "
            "(set_miscoverage uses softmax sets, undefined for scalar margins)."
        )
    n = labels.shape[0]
    m = lambdas.shape[0]
    losses_mat = np.zeros((n, m), dtype=float)
    stopped_mat = np.zeros((n, m), dtype=bool)
    stop_idx_mat = np.zeros((n, m), dtype=int)
    base = np.asarray(baseline_t, dtype=np.float64)
    for j, lam in enumerate(lambdas):
        thresholds_t = base * float(lam)
        stop_indices, stopped = stop_by_scalar_margin_ge(scores, thresholds_t)
        losses_mat[:, j] = argmax_miscoverage_losses_from_stops(
            probs, labels, stop_indices, stopped
        )
        stopped_mat[:, j] = stopped
        stop_idx_mat[:, j] = stop_indices
    return losses_mat, stopped_mat, stop_idx_mat


def fst_curve_scalar_margin(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    scores: np.ndarray,
    lambda_hat: float,
    baseline_t: np.ndarray,
) -> Dict[str, List[float]]:
    thresholds_t = np.asarray(baseline_t, dtype=np.float64) * float(lambda_hat)
    stop_indices, stopped = stop_by_scalar_margin_ge(scores, thresholds_t)
    t_steps = test_probs.shape[1]
    stopped_counts: List[int] = []
    cumulative_acc: List[float] = []
    for t in range(t_steps):
        stopped_by_t = stopped & (stop_indices <= t)
        count = int(np.sum(stopped_by_t))
        stopped_counts.append(count)
        if count == 0:
            cumulative_acc.append(0.0)
            continue
        idx = np.where(stopped_by_t)[0]
        t_stop = stop_indices[idx]
        preds = np.argmax(test_probs[idx, t_stop, :], axis=1)
        cumulative_acc.append(float(np.mean(preds == test_labels[idx])))
    return {"stopped_count": stopped_counts, "accuracy": cumulative_acc}


def lambda_method_curves_scalar(
    lambda_hat: Optional[float],
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    scores: np.ndarray,
    baseline_t: np.ndarray,
    t_steps: int,
) -> Dict[str, List[float]]:
    if lambda_hat is None:
        return {"stopped_count": [0] * t_steps, "accuracy": [0.0] * t_steps}
    return fst_curve_scalar_margin(
        test_probs,
        test_labels,
        scores,
        float(lambda_hat),
        baseline_t,
    )


def acquisition_rates_from_visible(
    visible_counts: np.ndarray,
    rate_denom: Optional[float],
) -> List[float]:
    vc = np.asarray(visible_counts, dtype=float).ravel()
    denom = float(vc.max()) if rate_denom is None else float(rate_denom)
    if denom <= 0:
        raise ValueError("rate denominator must be positive")
    return [float(c) / denom for c in vc.tolist()]


def jafa_natural_curves(
    test_labels: np.ndarray,
    predictions_full: np.ndarray,
    natural_stop_t: np.ndarray,
    frontier_original_col_indices: np.ndarray,
) -> Dict[str, List[float]]:
    """
    JAFA baseline: RL STOP at timestep natural_stop_t (indices into **full** export axis).

    `frontier_original_col_indices[tau]` is the original NPZ timestep index at coarse plot step
    `tau` (e.g. from --num_sampling_rates / --wanted_counts). Count sample i as stopped-by-tau
    iff natural_stop_t[i] <= that frontier — **not** iff natural_stop_t[i] <= tau, which breaks
    when trajectory columns are subsampled.
    """
    n_test = int(test_labels.shape[0])
    T_full = int(predictions_full.shape[1])
    labels_int = test_labels.astype(int, copy=False)
    t_stop_raw = natural_stop_t.astype(int, copy=False)
    t_stop_clipped = np.clip(t_stop_raw, 0, T_full - 1)
    preds_at_stop = predictions_full[np.arange(n_test), t_stop_clipped].astype(int, copy=False)

    frontier = np.asarray(frontier_original_col_indices, dtype=np.int64).ravel()
    t_steps = int(frontier.size)
    stopped_counts: List[int] = []
    cumulative_acc: List[float] = []
    for tau in range(t_steps):
        col_cap = frontier[tau]
        mask = (t_stop_raw >= 0) & (t_stop_raw <= col_cap)
        count = int(np.sum(mask))
        stopped_counts.append(count)
        if count == 0:
            cumulative_acc.append(0.0)
            continue
        idx = np.where(mask)[0]
        preds = preds_at_stop[idx]
        cumulative_acc.append(float(np.mean(preds == labels_int[idx])))
    return {"stopped_count": stopped_counts, "accuracy": cumulative_acc}


def lambda_method_curves_compare(
    lambda_hat: Optional[float],
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    stopping_rule: str,
    baseline_t: np.ndarray,
    set_size_max: int,
    t_steps: int,
    n_test: int,
) -> Dict[str, List[float]]:
    if lambda_hat is None:
        return {
            "stopped_count": [0] * t_steps,
            "accuracy": [0.0] * t_steps,
        }
    return fst_curve(
        test_probs,
        test_labels,
        float(lambda_hat),
        stopping_rule,
        baseline_t,
        set_size_max,
    )


def plot_rl_compare_stopped(
    output_dir: Path,
    acquisition_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_stopped_count: List[int],
    plot_tag: str,
    n_test: int,
) -> None:
    inv = 1.0 / float(n_test)
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)
    order_keys = list(ORDER_KEYS_RL)
    for key in order_keys:
        frac = [int(c) * inv for c in curves_by_method[key]["stopped_count"]]
        ax.plot(
            acquisition_rates,
            frac,
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Fraction of test samples stopped")
    ax.set_ylim(0.0, 1.05)
    style_compare_ax(ax, grid=True)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_rl_jafa_stopped_vs_rate_{plot_tag}.png"
    save_compare_figure(fig, out_path)


def plot_rl_compare_accuracy(
    output_dir: Path,
    acquisition_rates: List[float],
    accuracy_by_method: Dict[str, Dict[str, List[float]]],
    plot_tag: str,
    alpha_control: float,
    loss_kind: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)
    if loss_kind == "argmax_miscoverage":
        target_acc = float(max(0.0, min(1.0, 1.0 - alpha_control)))
        ax.axhline(
            target_acc,
            linestyle="--",
            linewidth=0.9,
            color="gray",
            zorder=0,
            alpha=0.85,
            label=f"reference 1−α = {target_acc:g}",
        )
    order_keys = list(ORDER_KEYS_RL)
    for key in order_keys:
        ax.plot(
            acquisition_rates,
            accuracy_by_method[key]["accuracy"],
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Accuracy of stopped samples")
    style_compare_ax(ax, grid=True)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_rl_jafa_stopped_accuracy_vs_rate_{plot_tag}.png"
    save_compare_figure(fig, out_path)


def validate_bundle(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    natural_stop_t: Optional[np.ndarray],
    predictions: np.ndarray,
    visible_counts: np.ndarray,
):
    probs = np.asarray(probs)
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions)
    vc = np.asarray(visible_counts).ravel()
    t_steps_probs = probs.shape[1]
    if vc.size != t_steps_probs:
        raise ValueError(f"{name}: visible_counts length {vc.size} != probs T {t_steps_probs}")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError(f"{name}: labels vs probs batch mismatch")
    if predictions.shape != probs.shape[:2]:
        raise ValueError(f"{name}: predictions shape {predictions.shape} vs probs [:2]")
    if natural_stop_t is not None:
        nt = np.asarray(natural_stop_t).ravel()
        if nt.shape[0] != labels.shape[0]:
            raise ValueError(f"{name}: natural_stop_t length mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RL JAFA trajectory comparison: JAFA stopping vs calibrated wrappers "
            "(same stopping rule as eval_compare_cifar10 on prob trajectories)."
        )
    )
    parser.add_argument(
        "--val_scores_npz",
        type=str,
        default=None,
        help="NPZ path for val split; not needed if --score_dir is set",
    )
    parser.add_argument(
        "--test_scores_npz",
        type=str,
        default=None,
        help="NPZ path for test split; not needed if --score_dir is set",
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default="/gpfs/scratch/td2105/rl_cifar_score_matrices/cifar_best",
        help="Loads <score_dir>/val_scores.npz and test_scores.npz",
    )
    parser.add_argument(
        "--wanted_counts",
        type=str,
        default="",
        help="Comma-separated visible_counts columns to keep. Mutually exclusive with --num_sampling_rates.",
    )
    parser.add_argument(
        "--num_sampling_rates",
        type=int,
        default=10,
        help=(
            "If set (and --wanted_counts is empty), uniformly subsample this many trajectory "
            "columns over the NPZ timestep axis (fewer ⇒ fewer Casc steps ⇒ α split across "
            "fewer checkpoints). Example: 10 with a 61-step export yields ~evenly spaced depths."
        ),
    )
    parser.add_argument(
        "--rate_denom",
        type=float,
        default=None,
        help="Denominator for x-axis rate=max_visible/denom (default visible_counts.max())",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--est_size", type=int, default=100)
    parser.add_argument(
        "--calib_fraction",
        type=float,
        default=1.0,
        help="Fraction of val-after-est rows used for calibration. 1.0 = use all remainder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./neurips_plots_final/cifar_rl_jafa_compare",
        help=(
            "Base output directory; artifacts are written under "
            "<output_dir>/max-prob/, q-margin-mean/, or q-margin-max/ from --wrapper_signal."
        ),
    )
    parser.add_argument(
        "--score_cache_base",
        type=str,
        default=None,
        help=(
            "Unused: this script does not use a score_cache directory. "
            "Accepted for CLI parity with other eval_compare_* scripts."
        ),
    )
    parser.add_argument("--num_lambdas", type=int, default=81)
    parser.add_argument(
        "--stopping_rule",
        type=str,
        default="max_score",
        choices=["max_score", "set_size"],
        help=(
            "Used only when --wrapper_signal probs (max_score vs prediction-set size); "
            "ignored for q_margin_* (margins always use score >= λ * baseline)."
        ),
    )
    parser.add_argument("--set_size_max", type=int, default=1)
    parser.add_argument("--delta_est", type=float, default=0.05)
    parser.add_argument(
        "--wrapper_signal",
        type=str,
        default="q_margin_max",
        choices=sorted(WRAPPER_SIGNAL_NPZ_KEYS.keys()),
        help=(
            "probs: same as eval_compare_cifar10. q_margin_*: NPZ score_q_margin_max / "
            "score_q_margin_mean thresholded stops; requires loss_kind argmax_miscoverage "
            "and stopping_rule max_score (λ grid 1→0)."
        ),
    )
    parser.add_argument(
        "--casc_alpha_schedule",
        type=str,
        default="uniform",
        choices=["uniform", "linear_increasing", "linear_decreasing", "geometric"],
    )
    parser.add_argument("--abstain_penalty", type=float, default=0.01)
    parser.add_argument(
        "--ltt_abstain_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
    )
    parser.add_argument(
        "--ltt_conditional_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
    )
    parser.add_argument("--default_lambda_idx_abstain", type=int, default=None)
    parser.add_argument("--default_lambda_idx_conditional", type=int, default=None)
    parser.add_argument(
        "--loss_kind",
        type=str,
        default="argmax_miscoverage",
        choices=["set_miscoverage", "argmax_miscoverage"],
    )
    parser.add_argument(
        "--plot_forced",
        action="store_true",
        help="Also plot forced frontier prediction accuracy (every sample classified at depth t)",
    )
    parser.add_argument(
        "--plot_conditional",
        action="store_true",
        help=(
            "Second column is already stopped-sample accuracy; appends _conditional to panel PNG "
            "(for naming consistency with three-column FNR/FPR compares)."
        ),
    )
    args = parser.parse_args()
    configure_compare_plot_style()

    if args.score_dir:
        sd = Path(args.score_dir)
        val_npz_path = sd / "val_scores.npz"
        test_npz_path = sd / "test_scores.npz"
    elif args.val_scores_npz and args.test_scores_npz:
        val_npz_path = Path(args.val_scores_npz)
        test_npz_path = Path(args.test_scores_npz)
    else:
        raise SystemExit(
            "Provide --score_dir or both --val_scores_npz and --test_scores_npz"
        )

    val_raw = load_scores_npz(val_npz_path)
    test_raw = load_scores_npz(test_npz_path)
    md_path = val_npz_path.parent / "metadata.json"
    meta = load_optional_json(md_path)

    for key in ("probs", "labels", "predictions", "visible_counts"):
        if key not in val_raw:
            raise KeyError(f"val_scores_npz missing key {key!r}")

    vc_full = np.asarray(val_raw["visible_counts"]).ravel()
    wanted_parts = [p.strip() for p in args.wanted_counts.split(",") if p.strip()]
    if wanted_parts:
        if args.num_sampling_rates is not None:
            raise SystemExit("Use either --wanted_counts or --num_sampling_rates, not both.")
        wanted_list = [int(x) for x in wanted_parts]
        col_idx = column_indices_for_visible_counts(vc_full, wanted_list)
    elif args.num_sampling_rates is not None:
        col_idx = uniform_subsample_column_indices(vc_full.size, args.num_sampling_rates)
        print(
            f"Subsampled {col_idx.size} trajectory columns "
            f"(requested --num_sampling_rates={args.num_sampling_rates}); "
            f"visible_counts={np.asarray(vc_full.ravel())[col_idx].tolist()}"
        )
    else:
        col_idx = np.arange(vc_full.size, dtype=np.int64)

    val_b = slice_trajectory_bundle(val_raw, col_idx)
    test_b = slice_trajectory_bundle(test_raw, col_idx)

    pred_test_full = np.asarray(test_raw["predictions"])
    full_t_expect = int(vc_full.size)
    if pred_test_full.shape[1] != full_t_expect:
        raise ValueError(
            f"predictions timeline {pred_test_full.shape[1]} != visible_counts length {full_t_expect}"
        )

    if "natural_stop_t" not in test_b:
        raise KeyError(
            "test_scores_npz must contain natural_stop_t for JAFA baseline (see RL export)."
        )

    probs_val = np.asarray(val_b["probs"], dtype=np.float32)
    probs_test = np.asarray(test_b["probs"], dtype=np.float32)
    labels_val = np.asarray(val_b["labels"]).ravel()
    labels_test = np.asarray(test_b["labels"]).ravel()
    pred_test = np.asarray(test_b["predictions"])

    vc = np.asarray(val_b["visible_counts"]).ravel()
    if not np.array_equal(np.asarray(test_b["visible_counts"]).ravel(), vc):
        raise ValueError("val/test visible_counts subsets must agree after slicing")

    validate_bundle(
        "val",
        probs_val,
        labels_val,
        None,
        np.asarray(val_b["predictions"]),
        vc,
    )
    nt_test = np.asarray(test_b["natural_stop_t"]).ravel()
    validate_bundle(
        "test",
        probs_test,
        labels_test,
        nt_test,
        pred_test,
        vc,
    )

    margin_npz_key = WRAPPER_SIGNAL_NPZ_KEYS[args.wrapper_signal]
    scores_val = scores_test_margin = None
    if margin_npz_key is not None:
        if margin_npz_key not in val_raw or margin_npz_key not in test_raw:
            raise KeyError(
                f"wrapper_signal={args.wrapper_signal} requires NPZ keys {margin_npz_key!r}"
            )
        scores_val = np.asarray(val_b[margin_npz_key], dtype=np.float64)
        scores_test_margin = np.asarray(test_b[margin_npz_key], dtype=np.float64)
        if scores_val.shape != probs_val.shape[:2]:
            raise ValueError(
                f"{margin_npz_key} val shape {scores_val.shape} != probs [:2]"
            )
        if scores_test_margin.shape != probs_test.shape[:2]:
            raise ValueError(
                f"{margin_npz_key} test shape {scores_test_margin.shape} != probs [:2]"
            )

    n_val = probs_val.shape[0]
    use_full_calib = args.calib_fraction >= 1.0 - 1e-12
    est_idx, calib_idx = split_val_into_est_calib(
        n_val,
        args.est_size,
        args.calib_fraction,
        args.seed,
        use_full_remaining_as_calib=use_full_calib,
    )

    est_probs = probs_val[est_idx]
    est_labels = labels_val[est_idx]
    calib_probs = probs_val[calib_idx]
    calib_labels = labels_val[calib_idx]

    t_steps = probs_test.shape[1]
    n_test = int(labels_test.shape[0])

    use_margin_scores = args.wrapper_signal != "probs"
    if use_margin_scores:
        if args.stopping_rule != "max_score":
            raise ValueError(
                "wrapper_signal q_margin_* requires --stopping_rule max_score "
                "(λ grid shrinks thresholds on margins; use probs + set_size for set stopping)."
            )
        if args.loss_kind != "argmax_miscoverage":
            raise ValueError(
                "wrapper_signal q_margin_* requires --loss_kind argmax_miscoverage"
            )

    labels_test_int = labels_test.astype(int, copy=False)

    if use_margin_scores:
        if scores_val is None or scores_test_margin is None:
            raise RuntimeError("internal: margin scores not loaded")
        est_scores = scores_val[est_idx]
        baseline_t = build_scalar_margin_baseline(est_scores, args.delta_est).astype(np.float64)
        lambdas = np.linspace(1.0, 0.0, args.num_lambdas)
        if baseline_t.shape[0] != t_steps:
            raise ValueError("Internal error: baseline_t length mismatches trajectory T")

        calib_scores = scores_val[calib_idx]
        losses_mat, stopped_mat, stop_idx_mat = per_lambda_calibration_scalar_margin(
            calib_probs,
            calib_labels.astype(int),
            calib_scores,
            lambdas,
            baseline_t,
            args.loss_kind,
        )

        method_selections = select_lambda_methods(
            lambdas,
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
        )

        curves_other = {}
        for key in ["ltt_conditional"]:
            curves_other[key] = lambda_method_curves_scalar(
                method_selections[key]["lambda_hat"],
                probs_test,
                labels_test_int,
                scores_test_margin,
                baseline_t,
                t_steps,
            )

    else:
        baseline_t = (
            build_max_score_baseline(est_probs, args.delta_est)
            if args.stopping_rule == "max_score"
            else build_true_class_baseline(est_probs, est_labels.astype(int), args.delta_est)
        )
        if baseline_t.shape[0] != t_steps:
            raise ValueError("Internal error: baseline_t length mismatches probs T")

        lambdas = (
            np.linspace(1.0, 0.0, args.num_lambdas)
            if args.stopping_rule == "max_score"
            else np.linspace(0.0, 1.0, args.num_lambdas)
        )

        losses_mat, stopped_mat, stop_idx_mat = per_lambda_calibration_arrays(
            calib_probs,
            calib_labels.astype(int),
            lambdas,
            args.stopping_rule,
            baseline_t,
            args.set_size_max,
            loss_kind=args.loss_kind,
        )

        method_selections = select_lambda_methods(
            lambdas,
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
        )

        curves_other = {}
        for key in ["ltt_conditional"]:
            curves_other[key] = lambda_method_curves_compare(
                method_selections[key]["lambda_hat"],
                probs_test,
                labels_test_int,
                args.stopping_rule,
                baseline_t,
                args.set_size_max,
                t_steps,
                n_test,
            )

    plot_tag = f"{args.stopping_rule}_{args.wrapper_signal}"

    casc_stats = casc_curve(
        calib_probs,
        calib_labels.astype(int),
        probs_test,
        labels_test_int,
        args.alpha,
        args.casc_alpha_schedule,
    )

    curves_by_method = {
        "jafa": jafa_natural_curves(
            labels_test_int, pred_test_full, nt_test, col_idx
        ),
        **curves_other,
        "casc": casc_stats,
    }

    forced_stats = forced_curve(probs_test, labels_test_int)

    acquisition_rates = acquisition_rates_from_visible(vc, args.rate_denom)

    score_type_subdir = WRAPPER_SIGNAL_OUTPUT_SUBDIR[args.wrapper_signal]
    out_dir = Path(args.output_dir) / score_type_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (score type {score_type_subdir}): {out_dir}")

    accuracy_by_method = {
        "jafa": curves_by_method["jafa"],
        **curves_other,
        "casc": casc_stats,
    }

    plot_rl_compare_panel(
        out_dir,
        acquisition_rates,
        curves_by_method,
        accuracy_by_method,
        plot_tag,
        n_test,
        args.alpha,
        args.loss_kind,
        plot_conditional=bool(args.plot_conditional),
    )

    out_summary = {
        "sources": {"val_npz": str(val_npz_path), "test_npz": str(test_npz_path)},
        "plot_conditional": bool(args.plot_conditional),
        "output_base_dir": str(Path(args.output_dir)),
        "output_score_type_subdir": score_type_subdir,
        "artifacts_dir": str(out_dir),
        "visible_counts_used": vc.tolist(),
        "trajectory_selection": (
            {"mode": "wanted_counts_list", "wanted_counts_csv": args.wanted_counts.strip()}
            if wanted_parts
            else {
                "mode": "uniform_subsample",
                "num_sampling_rates_requested": args.num_sampling_rates,
            }
            if args.num_sampling_rates is not None
            else {"mode": "all_columns", "full_timesteps": int(vc_full.size)}
        ),
        "acquisition_rates": acquisition_rates,
        "metadata_probe": meta,
        "alpha": args.alpha,
        "delta": args.delta,
        "n_val": n_val,
        "n_est": int(est_probs.shape[0]),
        "n_cal": int(calib_probs.shape[0]),
        "n_test": n_test,
        "wrapper_signal": args.wrapper_signal,
        "margin_npz_key": margin_npz_key,
        "stopping_rule": args.stopping_rule,
        "set_size_max": args.set_size_max,
        "delta_est": args.delta_est,
        "casc_alpha_schedule": args.casc_alpha_schedule,
        "loss_kind": args.loss_kind,
        "seed": args.seed,
        "est_indices_count": int(est_idx.size),
        "calib_indices_count": int(calib_idx.size),
        "abstain_penalty": float(args.abstain_penalty),
        "ltt_abstain_utility": args.ltt_abstain_utility,
        "ltt_conditional_utility": args.ltt_conditional_utility,
        "selections": {
            key: {
                "lambda_hat": (
                    None if sel["lambda_hat"] is None else float(sel["lambda_hat"])
                ),
                "reason": sel["meta"]["reason"],
            }
            for key, sel in method_selections.items()
        },
        "curves": {
            **{k: curves_by_method[k] for k in ["jafa", "ltt_conditional"]},
            "bonferroni": {
                k: casc_stats[k]
                for k in ("stopped_count", "accuracy", "alpha_k", "alpha_k_sum")
                if k in casc_stats
            },
            "forced": forced_stats if args.plot_forced else None,
        },
    }

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    out_path = out_dir / f"compare_rl_jafa_summary_{plot_tag}.json"
    with open(out_path, "w") as handle:
        json.dump(out_summary, handle, indent=2, default=_json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
