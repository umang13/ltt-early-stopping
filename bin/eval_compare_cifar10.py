#!/usr/bin/env python3
"""
CIFAR-10 comparison: **LTT-conditional** (Holm) and forced baseline.

Single calibration/test split with a shared stopping rule; one row of two subplots and shared legend:
  - stopped fraction vs sampling rate
  - stopped-sample accuracy vs sampling rate (incl. Forced)

Calibration tensors match legacy triple-method setups; only **LTT-cond** λ selection remains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

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

from eval_fst_cifar10 import (
    build_max_score_baseline,
    build_true_class_baseline,
    build_score_cache_dir,
    casc_curve,
    collect_prob_tensor,
    collect_prob_tensor_cached,
    fixed_threshold_multiclass_schedule_curves,
    forced_curve,
    fst_curve,
    get_val_test_datasets,
    load_model,
    per_lambda_calibration_arrays,
    split_dataset,
    split_estimation_set,
)
from classifier.utils.utils import get_config

from fst.ltt import (
    compute_grid_pvalues,
    grid_risks_and_ns,
    grid_utilities,
    ltt_select_lambda,
)


METHOD_COLORS = {
    "ltt_conditional": "#8172B2",
    "casc": "#009E73",
    "forced": "#4A4A4A",
    # Stronger hues than the MRI chartreuse/cyan pair (hard to read on #F7F8FA).
    "confidence_threshold": "#D55E00",
    "entropy_threshold": "#0072B2",
}

METHOD_LABELS = {
    "ltt_conditional": COMPARE_RCS_LEGEND_LABEL,
    "casc": "Bonferroni",
    "forced": "Forced",
    "confidence_threshold": "Confidence threshold",
    "entropy_threshold": "Entropy threshold",
}

METHOD_LINESTYLES = {
    "ltt_conditional": "--",
    "casc": "-",
    "forced": "-",
    "confidence_threshold": ":",
    "entropy_threshold": ":",
}

METHOD_MARKERS = {
    "ltt_conditional": "D",
    "casc": "o",
    "forced": "o",
    "confidence_threshold": "v",
    "entropy_threshold": "^",
}


def _merge_legend_by_priority_cifar(
    axes: List[Any],
    label_priority: List[str],
) -> Tuple[List, List]:
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


def plot_compare_cifar10_panel(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_stats: Dict[str, List[float]],
    forced_stats: Dict[str, List[float]],
    stopping_rule: str,
    n_test: int,
    alpha_control: float,
    loss_kind: str,
    plot_conditional: bool = False,
    *,
    confidence_threshold_curve: Optional[Dict[str, List[float]]] = None,
    confidence_threshold_legend: Optional[str] = None,
    entropy_threshold_curve: Optional[Dict[str, List[float]]] = None,
    entropy_threshold_legend: Optional[str] = None,
) -> None:
    """One row, two subplots; shared legend above (stopped fraction + stopped-sample accuracy)."""
    table_rows: List[Tuple[str, Dict[str, List[float]]]] = [
        (METHOD_LABELS["ltt_conditional"], curves_by_method["ltt_conditional"]),
        (METHOD_LABELS["forced"], forced_stats),
    ]
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
    print_compare_curve_table(
        sampling_rates,
        table_rows,
        metric_key="accuracy",
        metric_label="stopped-sample accuracy",
        title=(
            f"Plot data table [cifar10_{stopping_rule}"
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
            sampling_rates,
            ys,
            color=METHOD_COLORS[key],
            linestyle=METHOD_LINESTYLES[key],
            marker=METHOD_MARKERS[key],
            label=lab,
            zorder=zorder,
        )

    for key in ["ltt_conditional"]:
        _series(
            ax0,
            key,
            [c * inv for c in curves_by_method[key]["stopped_count"]],
            METHOD_LABELS[key],
            zorder=3,
        )
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        _series(
            ax0,
            "confidence_threshold",
            [c * inv for c in confidence_threshold_curve["stopped_count"]],
            lab_c,
        )
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        _series(
            ax0,
            "entropy_threshold",
            [c * inv for c in entropy_threshold_curve["stopped_count"]],
            lab_e,
        )
    ax0.set_ylabel("Fraction stopped")
    ax0.set_ylim(0.0, 1.05)
    style_compare_ax(ax0, grid=True)
    ax0.set_xlabel(COMPARE_XLABEL_SAMPLING)

    target_acc = float(max(0.0, min(1.0, 1.0 - alpha_control)))
    prio = [
        METHOD_LABELS["ltt_conditional"],
        METHOD_LABELS["forced"],
    ]
    if confidence_threshold_legend:
        prio.append(confidence_threshold_legend)
    elif confidence_threshold_curve is not None:
        prio.append(METHOD_LABELS["confidence_threshold"])
    if entropy_threshold_legend:
        prio.append(entropy_threshold_legend)
    elif entropy_threshold_curve is not None:
        prio.append(METHOD_LABELS["entropy_threshold"])
    if loss_kind == "argmax_miscoverage":
        ref_lab = f"1−α = {target_acc:g}"
        draw_accuracy_target_reference(ax1, target_acc, label=ref_lab)
        prio = [ref_lab] + prio

    for key in ["ltt_conditional"]:
        _series(ax1, key, curves_by_method[key]["accuracy"], METHOD_LABELS[key], zorder=3)
    _series(ax1, "forced", forced_stats["accuracy"], METHOD_LABELS["forced"], zorder=1)
    if confidence_threshold_curve is not None:
        lab_c = confidence_threshold_legend or METHOD_LABELS["confidence_threshold"]
        _series(ax1, "confidence_threshold", confidence_threshold_curve["accuracy"], lab_c)
    if entropy_threshold_curve is not None:
        lab_e = entropy_threshold_legend or METHOD_LABELS["entropy_threshold"]
        _series(ax1, "entropy_threshold", entropy_threshold_curve["accuracy"], lab_e)
    ax1.set_ylabel("Accuracy of stopped samples")
    style_compare_ax(ax1, grid=True)
    ax1.set_xlabel(COMPARE_XLABEL_SAMPLING)

    handles, labels = _merge_legend_by_priority_cifar([ax0, ax1], prio)
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
    stem = f"compare_panel_{stopping_rule}"
    if plot_conditional:
        stem += "_conditional"
    out_path = output_dir / f"{stem}.png"
    save_compare_figure(fig, out_path, dpi=300)


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
) -> Dict[str, Dict]:
    """
    Returns {method: {"lambda_hat", "meta", "risks", "ns", "p_values", "utilities"}}
    for **ltt_conditional** only.

    ``abstain_penalty``, ``ltt_abstain_utility``, ``default_lambda_idx_abstain`` are ignored (CLI compat).
    """
    risks_cond, ns_cond = grid_risks_and_ns(
        losses_mat, stopped_mat, "conditional_on_stopped", 0.0
    )
    p_cond = compute_grid_pvalues(risks_cond, ns_cond, alpha, B=1.0)
    util_cond = grid_utilities(
        stopped_mat, stop_idx_mat, risks_cond, ltt_conditional_utility
    )

    ltt_conditional_lam, ltt_conditional_meta = ltt_select_lambda(
        lambdas, risks_cond, p_cond, delta, util_cond,
        default_lambda_idx=default_lambda_idx_conditional,
    )

    return {
        "ltt_conditional": {
            "lambda_hat": ltt_conditional_lam,
            "meta": ltt_conditional_meta,
            "risks": risks_cond.tolist(),
            "ns": ns_cond.tolist(),
            "p_values": p_cond.tolist(),
            "utilities": util_cond.tolist(),
        },
    }


def lambda_method_curves(
    lambda_hat: Optional[float],
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    stopping_rule: str,
    baseline_t: np.ndarray,
    set_size_max: int,
    n_test: int,
    t_steps: int,
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


def plot_compare_stopped(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_stopped_count: List[int],
    stopping_rule: str,
    n_test: int,
) -> None:
    if n_test <= 0:
        raise ValueError("n_test must be positive for stopped fraction plots")
    inv = 1.0 / float(n_test)
    fig, ax = plt.subplots(1, 1, figsize=COMPARE_FIGSIZE_SINGLE, dpi=300)
    for key in ["ltt_conditional"]:
        frac = [c * inv for c in curves_by_method[key]["stopped_count"]]
        ax.plot(
            sampling_rates,
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
    out_path = output_dir / f"compare_stopped_vs_sampling_rate_{stopping_rule}.png"
    save_compare_figure(fig, out_path)


def plot_compare_stopped_accuracy(
    output_dir: Path,
    sampling_rates: List[float],
    curves_by_method: Dict[str, Dict[str, List[float]]],
    casc_stats: Dict[str, List[float]],
    forced_stats: Dict[str, List[float]],
    stopping_rule: str,
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
    for key in ["ltt_conditional"]:
        ax.plot(
            sampling_rates,
            curves_by_method[key]["accuracy"],
            marker=METHOD_MARKERS[key],
            linestyle=METHOD_LINESTYLES[key],
            color=METHOD_COLORS[key],
            label=METHOD_LABELS[key],
        )
    ax.plot(
        sampling_rates,
        forced_stats["accuracy"],
        marker=METHOD_MARKERS["forced"],
        linestyle=METHOD_LINESTYLES["forced"],
        color=METHOD_COLORS["forced"],
        label=METHOD_LABELS["forced"],
    )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Accuracy of stopped samples")
    style_compare_ax(ax, grid=True)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"compare_stopped_accuracy_vs_sampling_rate_{stopping_rule}.png"
    save_compare_figure(fig, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 comparison: LTT-conditional (Holm) vs forced."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/partial/cifar10_bayes_biased_random_rate_config.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/gpfs/data/chopralab/us453/icml_26/paper_release/results/checkpoints/cifar10/bayes_biased_random_rate/cifar10_baseline_epoch=49_val_accuracy=0.7238.ckpt",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Target risk level for H0: R > alpha.")
    parser.add_argument("--delta", type=float, default=0.05, help="FWER level for FST/LTT.")
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--est_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./neurips_plots_final/cifar10_compare")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sampling_rates",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--num_lambdas", type=int, default=101)
    parser.add_argument(
        "--stopping_rule",
        type=str,
        default="max_score",
        choices=["max_score", "set_size"],
        help="Decision function D_lambda(x_t); shared calibration grid for LTT-cond.",
    )
    parser.add_argument("--set_size_max", type=int, default=1)
    parser.add_argument(
        "--delta_est",
        type=float,
        default=0.05,
        help="Quantile level on estimation set for per-time baseline thresholds.",
    )
    parser.add_argument(
        "--casc_alpha_schedule",
        type=str,
        default="uniform",
        choices=["uniform", "linear_increasing", "linear_decreasing", "geometric"],
    )
    parser.add_argument(
        "--cumulative_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--abstain_penalty",
        type=float,
        default=1,
        help="Loss assigned to non-stopped (abstained) calibration samples for LTT-abstain.",
    )
    parser.add_argument(
        "--ltt_abstain_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
        help="Utility for LTT with abstention cost (default: minimize overall risk).",
    )
    parser.add_argument(
        "--ltt_conditional_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
        help="Utility for LTT with risk conditional on stopped (default: max stopped count).",
    )
    parser.add_argument("--default_lambda_idx_abstain", type=int, default=None)
    parser.add_argument("--default_lambda_idx_conditional", type=int, default=None)
    parser.add_argument(
        "--loss_kind",
        type=str,
        default="argmax_miscoverage",
        choices=["set_miscoverage", "argmax_miscoverage"],
        help=(
            "Per-sample calibration loss. Default 'argmax_miscoverage' aligns the "
            "calibration risk with the test-time argmax accuracy plotted here, and "
            "avoids the trivial set-coverage degeneracy at small lambda. Pass "
            "'set_miscoverage' to match the legacy eval_fst_cifar10.py behavior."
        ),
    )
    parser.add_argument(
        "--force_recompute_scores",
        action="store_true",
        help="Recompute probabilities/labels and overwrite the score cache.",
    )
    parser.add_argument(
        "--score_cache_base",
        type=str,
        default="/gpfs/data/chopralab/us453/icml_26/paper_release/neurips_plots/cifar10_compare/testset",
        help=(
            "If set, read/write score caches under <dir>/score_cache/seed_<seed>/ instead of under "
            "the resolved output directory (after testset/). Same layout as the default cache tree."
        ),
    )
    parser.add_argument(
        "--plot_conditional",
        action="store_true",
        help=(
            "This two-column panel already plots stopped-sample accuracy; the flag only appends "
            "_conditional to the PNG filename for consistent naming with FastMRI/ImageNet compares."
        ),
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Optional baseline: stop at first rate where max class probability ≥ this value.",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.5,
        help="Optional baseline: stop at first rate where row entropy ≤ this value.",
    )
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

    val_ds, test_ds = get_val_test_datasets(config, args.use_testset)
    est_ds, remaining = split_estimation_set(val_ds, args.est_size, args.seed)
    if args.use_testset and test_ds is not None:
        calib_ds = remaining
    else:
        calib_ds, test_ds = split_dataset(remaining, args.calib_fraction, args.seed)

    batch_size = args.batch_size or getattr(config, "val_batch_size", getattr(config, "batch_size", 32))
    model = load_model(config, args.checkpoint_path, device)
    sampling_rates = [float(x) for x in args.sampling_rates.split(",") if x.strip()]

    out_dir = Path(args.output_dir)
    if args.use_testset:
        out_dir = out_dir / "testset"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.score_cache_base).expanduser() if args.score_cache_base else out_dir
    cache_dir = build_score_cache_dir(str(cache_root), args.seed)
    base_cache_key: Dict = {
        "seed": int(args.seed),
        "calib_fraction": float(args.calib_fraction),
        "est_size": int(args.est_size),
        "use_testset": bool(args.use_testset),
        "cumulative_sampling": bool(args.cumulative_sampling),
        "sampling_rates": tuple(sampling_rates),
        "checkpoint_path": str(args.checkpoint_path),
        "label_names": tuple(label_names),
    }

    est_probs, est_labels = collect_prob_tensor_cached(
        model, config, est_ds, "est", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "est", "dataset_size": int(len(est_ds))},
        args.force_recompute_scores,
    )
    calib_probs, calib_labels = collect_prob_tensor_cached(
        model, config, calib_ds, "calib", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "calib", "dataset_size": int(len(calib_ds))},
        args.force_recompute_scores,
    )
    test_probs, test_labels = collect_prob_tensor_cached(
        model, config, test_ds, "test", sampling_rates, label_names, device,
        batch_size, args.num_workers, args.cumulative_sampling,
        cache_dir,
        {**base_cache_key, "split": "test", "dataset_size": int(len(test_ds))},
        args.force_recompute_scores,
    )

    if args.stopping_rule == "max_score":
        baseline_t = build_max_score_baseline(est_probs, args.delta_est)
        lambdas = np.linspace(1.0, 0.0, args.num_lambdas)  # conservative -> aggressive
    else:
        baseline_t = build_true_class_baseline(est_probs, est_labels.astype(int), args.delta_est)
        lambdas = np.linspace(0.0, 1.0, args.num_lambdas)  # conservative -> aggressive

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

    m_grid = int(lambdas.shape[0])
    holm_tightest = float(args.delta) / max(m_grid, 1)
    print(
        f"Grid m={m_grid}, delta={args.delta}. "
        f"LTT-Cond uses Holm; tightest per-step cutoff = delta/m = {holm_tightest:.3g}."
    )
    # Fingerprint of the LTT-conditional inputs. Two runs with identical
    # fingerprints MUST produce identical LTT-conditional output regardless
    # of --abstain_penalty (which only enters LTT-abstain).
    cond_fingerprint = hash(
        (
            args.alpha,
            args.delta,
            args.stopping_rule,
            args.set_size_max,
            args.loss_kind,
            args.num_lambdas,
            args.seed,
            args.est_size,
            args.delta_est,
            args.calib_fraction,
            args.ltt_conditional_utility,
            args.default_lambda_idx_conditional,
            int(np.asarray(losses_mat).tobytes().__hash__()),
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
            f"  min r_hat={risks_arr[j_min_r]:.4f} at lambda={lambdas[j_min_r]:.4f} "
            f"(n_used={int(ns_arr[j_min_r])}, p={p_arr[j_min_r]:.3g})"
        )
        print(
            f"  min p_value={p_arr[j_min_p]:.3g} at lambda={lambdas[j_min_p]:.4f} "
            f"(r_hat={risks_arr[j_min_p]:.4f}, n_used={int(ns_arr[j_min_p])})"
        )
        rejected = np.asarray(sel["meta"].get("rejected", []), dtype=bool)
        print(
            f"  Holm rejected {int(rejected.sum())} / {m_grid} "
            f"(tightest cutoff delta/m={holm_tightest:.3g})"
        )

    n_test = int(test_labels.shape[0])
    t_steps = test_probs.shape[1]
    test_labels_int = test_labels.astype(int)

    curves_by_method: Dict[str, Dict[str, List[float]]] = {}
    for key in ["ltt_conditional"]:
        curves_by_method[key] = lambda_method_curves(
            method_selections[key]["lambda_hat"],
            test_probs,
            test_labels_int,
            args.stopping_rule,
            baseline_t,
            args.set_size_max,
            n_test,
            t_steps,
        )

    casc_stats = casc_curve(
        calib_probs,
        calib_labels.astype(int),
        test_probs,
        test_labels_int,
        args.alpha,
        args.casc_alpha_schedule,
    )
    forced_stats = forced_curve(test_probs, test_labels_int)

    conf_thr_curve, ent_thr_curve = fixed_threshold_multiclass_schedule_curves(
        test_probs,
        test_labels_int,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
    )
    conf_thr_legend = (
        f"Confidence ≥ {float(args.confidence_threshold):g}"
        if conf_thr_curve is not None
        else None
    )
    ent_thr_legend = (
        f"Entropy ≤ {float(args.entropy_threshold):g}"
        if ent_thr_curve is not None
        else None
    )

    plot_compare_cifar10_panel(
        out_dir,
        sampling_rates,
        curves_by_method,
        casc_stats,
        forced_stats,
        args.stopping_rule,
        n_test,
        args.alpha,
        args.loss_kind,
        plot_conditional=bool(args.plot_conditional),
        confidence_threshold_curve=conf_thr_curve,
        confidence_threshold_legend=conf_thr_legend,
        entropy_threshold_curve=ent_thr_curve,
        entropy_threshold_legend=ent_thr_legend,
    )

    out_summary = {
        "alpha": args.alpha,
        "plot_conditional": bool(args.plot_conditional),
        "delta": args.delta,
        "n_cal": int(calib_labels.shape[0]),
        "n_est": int(est_labels.shape[0]),
        "n_test": n_test,
        "stopping_rule": args.stopping_rule,
        "set_size_max": args.set_size_max,
        "delta_est": args.delta_est,
        "casc_alpha_schedule": args.casc_alpha_schedule,
        "loss_kind": args.loss_kind,
        "confidence_threshold": args.confidence_threshold,
        "entropy_threshold": args.entropy_threshold,
        "score_cache_base": args.score_cache_base,
        "score_cache_dir": str(cache_dir),
        "abstain_penalty": float(args.abstain_penalty),
        "ltt_abstain_utility": args.ltt_abstain_utility,
        "ltt_conditional_utility": args.ltt_conditional_utility,
        "default_lambda_idx_abstain": args.default_lambda_idx_abstain,
        "default_lambda_idx_conditional": args.default_lambda_idx_conditional,
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
            **curves_by_method,
            "casc": casc_stats,
            "forced": forced_stats,
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

    out_path = out_dir / f"compare_cifar10_summary_{args.stopping_rule}.json"
    with open(out_path, "w") as handle:
        json.dump(out_summary, handle, indent=2, default=_json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
