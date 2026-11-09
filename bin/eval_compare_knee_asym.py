#!/usr/bin/env python3
"""
FastMRI knee comparison (**asymmetric τ-dilation**) — duplicate of ``eval_compare_knee.py``.

Same **LTT-conditional (Holm)** / forced pipeline as the symmetric script (Bonferroni/CASC is computed for JSON but not drawn) except curves use
``compute_bands_asym(τ; γ_lo, γ_hi)``. Defaults ``γ_lo = γ_hi = 1`` recover symmetric ``λ``.
Legacy FST / LTT-abstain selection is omitted (panels match ``eval_compare_knee``).

By default the controlled risk is FNR (matching eval_fst_knee.py); pass
--risk_metric fpr to instead control FPR. Both metrics are plotted regardless.

The output directory is split: when --use_testset is passed, all artifacts
(plots, JSON summary, score cache) go into <output_dir>/testset/ to keep
test-set runs from clobbering the val-only sweeps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from compare_figure_export import COMPARE_RCS_LEGEND_LABEL, configure_compare_plot_style

from eval_compare_knee import (
    _append_per_label_maps,
    _empty_per_label_curve_block,
    _new_per_label_accumulators,
    cal_tuned_forced_curves,
    casc_curves,
    cal_tuned_forced_legend_label,
    class_conditional_fnr_fpr_per_label_over_idx,
    class_conditional_fnr_fpr_per_label_unconditional,
    class_conditional_fnr_fpr_unconditional,
    confidence_baseline_legend_label,
    entropy_baseline_legend_label,
    fixed_threshold_schedule_curves,
    fnr_cap_constrained_forced_cal_curves,
    forced_curves,
    forced_fnr_cal_legend_label,
    forced_method_legend_label,
    macro_fnr_fpr_over_idx,
    parse_alpha_per_label,
    plot_compare_fastmri_panel,
    print_class_conditional_label_diagnostics,
    resolve_alpha_per_label,
    resolve_control_label_indices,
    resolve_controlled_label_names,
    select_lambda_methods,
)


METHOD_COLORS = {
    "ltt_conditional": "#8172B2",
    "nominal_bands": "#DD8452",
    "casc": "#55A868",
    "forced": "#7F7F7F",
    "cal_tuned_forced": "#B07C9E",
}

METHOD_LABELS = {
    "ltt_conditional": COMPARE_RCS_LEGEND_LABEL,
    "nominal_bands": "Nominal band (λ=1)",
    "casc": "Bonferroni",
    "forced": forced_method_legend_label(),
    "cal_tuned_forced": cal_tuned_forced_legend_label("balanced_accuracy"),
}

METHOD_LINESTYLES = {
    "ltt_conditional": "--",
    "nominal_bands": "-.",
    "casc": "-",
    "forced": "-",
    "cal_tuned_forced": "--",
}

METHOD_MARKERS = {
    "ltt_conditional": "D",
    "nominal_bands": "^",
    "casc": "o",
    "forced": "o",
    "cal_tuned_forced": "s",
}

LAMBDA_METHODS = ["ltt_conditional"]

def compute_bands_asym(
    tau: float,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    gamma_lower: float,
    gamma_upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effective ambiguity bounds indexed by scalar τ∈[0,1].

    Frozen ``b_pos``, ``b_neg`` from ``compute_bands`` on the estimation set::

        lower = τ^{γ_lo} · b_neg,

        upper = τ^{γ_hi} · b_pos + (1 − τ^{γ_hi}).

    ``γ_lo = γ_hi = 1`` ≡ ``compute_bounds(tau, b_pos, b_neg)``.
    Requires ``gamma_lower, gamma_upper > 0``.
    """
    tau_cl = float(np.clip(float(tau), 0.0, 1.0))
    gl = float(gamma_lower)
    gu = float(gamma_upper)
    if gl <= 0.0 or gu <= 0.0:
        raise ValueError(f"gamma_lower / gamma_upper must be positive; got {gl}, {gu}")
    f_lo = float(np.power(tau_cl, gl))
    g_hi = float(np.power(tau_cl, gu))
    lower = f_lo * b_neg
    upper = g_hi * b_pos + (1.0 - g_hi)
    return lower, upper


def fpr_losses_with_abstain_zero(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped: np.ndarray,
) -> np.ndarray:
    losses = np.zeros(labels.shape[0], dtype=float)
    if np.any(stopped):
        losses[stopped] = compute_sample_fpr_loss(preds[stopped], labels[stopped])
    return losses


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
    gamma_lower: float,
    gamma_upper: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like baseline knee metric but ambiguity uses ``compute_bands_asym``."""
    n_cal = labels.shape[0]
    m = lambda_candidates.shape[0]
    n_labels = int(labels.shape[1])
    losses_mat = np.zeros((n_cal, m), dtype=float)
    stopped_mat = np.zeros((n_cal, m), dtype=bool)
    stop_idx_mat = np.zeros((n_cal, m), dtype=int)
    preds_cube = np.zeros((n_cal, m, n_labels), dtype=int)
    for j, lam in enumerate(lambda_candidates):
        lower, upper = compute_bands_asym(
            float(lam),
            b_pos,
            b_neg,
            gamma_lower,
            gamma_upper,
        )
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


def lambda_method_curves(
    lambda_hat: Optional[float],
    test_pos_scores: np.ndarray,
    label_matrix_test: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    gamma_lower: float,
    gamma_upper: float,
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

    lower, upper = compute_bands_asym(
        float(lambda_hat),
        b_pos,
        b_neg,
        gamma_lower,
        gamma_upper,
    )
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



def main():
    parser = argparse.ArgumentParser(
        description=(
            "FastMRI knee comparison (asymmetric τ-dilation along ambiguity bands): duplicate of "
            "eval_compare_knee.py."
        )
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/kspace/knee_slice_bayes_distill.yaml",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/gpfs/data/chopralab/us453/kspace_classifier/checkpoints/knee_slice_bayes_distill/2xtvl5na/checkpoints/bayes_distill_arms-knee-resnet18-epoch=16-best_metric=0.91.ckpt",
    )
    parser.add_argument("--alpha", type=float, default=0.075, help="Target risk level α (broadcast unless --alpha_per_label).")
    parser.add_argument(
        "--alpha_per_label",
        type=str,
        default=None,
        help="Comma-separated per-label α_k (length K); overrides broadcasting of --alpha for LTT.",
    )
    parser.add_argument("--delta", type=float, default=0.05, help="FWER level for FST/LTT.")
    parser.add_argument("--num_deltas", type=int, default=101)
    parser.add_argument("--est_size", type=int, default=200)
    parser.add_argument("--min_pos", type=int, default=3)
    parser.add_argument("--positive_class_idx", type=int, default=1)
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./neurips_plots_final/knee_slice_compare_asym")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling_rates", type=str, default="0.05,0.08,0.125,0.15,0.25")
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--num_lambdas", type=int, default=101)
    parser.add_argument("--use_est", action="store_true")
    parser.add_argument(
        "--band_gamma_lower",
        type=float,
        default=1.0,
        help="Exponent γ_lo on τ: lower = τ^γ_lo · b_neg (must be > 0); 1 ⇒ symmetric dilation.",
    )
    parser.add_argument(
        "--band_gamma_upper",
        type=float,
        default=1.0,
        help="Exponent γ_hi on τ: upper = τ^γ_hi · b_pos + (1-τ^γ_hi); must be > 0.",
    )
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
        default="/gpfs/data/chopralab/us453/icml_26/paper_release/neurips_plots/knee_slice_compare_asym/testset",
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
        help="Per-sample loss controlled by FST/LTT: 'fnr' (default) or 'fpr'.",
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        default=None,
        help="Comma-separated labels to omit from Option (i) control (e.g. acl).",
    )
    parser.add_argument(
        "--control_labels",
        type=str,
        default=None,
        help="Comma-separated labels to include in Option (i) control (e.g. mtear,abnormal).",
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
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
        help="Utility for LTT with abstention cost (default: stop as many as possible).",
    )
    parser.add_argument(
        "--ltt_conditional_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
        help="Utility for LTT with risk conditional on stopped (default: stop as many as possible).",
    )
    parser.add_argument("--default_lambda_idx_abstain", type=int, default=None)
    parser.add_argument("--default_lambda_idx_conditional", type=int, default=None)
    parser.add_argument(
        "--ltt_conditional_test",
        type=str,
        default="ratio",
        choices=["ratio", "via_W"],
        help=(
            "LTT-conditional: 'ratio' = Hoeffding on conditional empirical mean with n=n_stopped "
            "(degenerate when few stops per λ on knee_vol); 'via_W' (default) = Hoeffding on "
            "W_i=S_i*(L_i-alpha) with n=n_cal fixed (recommended for knee volume)."
        ),
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
            "Add forced baseline with calibration FN-capped per-(t,k) θ "
            "(see --forced_fnr_cal_alpha, --forced_fnr_match_casc_schedule)."
        ),
    )
    parser.add_argument(
        "--forced_fnr_cal_alpha",
        type=float,
        default=None,
        help="FNR cap on cal positives when fitting that baseline (defaults to --alpha).",
    )
    parser.add_argument(
        "--forced_fnr_match_casc_schedule",
        action="store_true",
        help="Use α/T per timestep (Bonferroni-style split) instead of a fixed cap per t.",
    )
    parser.add_argument(
        "--forced_fnr_cal_grid",
        type=int,
        default=101,
        help="Uniformly spaced θ grid for FN-capped forced search.",
    )
    parser.add_argument(
        "--plot_conditional",
        action="store_true",
        help=(
            "Panel middle subplot: cumulative FNR/FPR on cumulative stopped cohort only "
            "(saves compare_panel_<tag>_conditional.png)."
        ),
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help=(
            "Optional baseline: each sample stops at the first rate where every label's "
            "positive score is ≥ this threshold."
        ),
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.2,
        help="Optional baseline: first rate where mean label entropy ≤ this value.",
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

    print(
        "[eval_compare_knee_asym] asymmetric τ dilation: γ_lo="
        f"{args.band_gamma_lower:g}, γ_hi={args.band_gamma_upper:g} "
        "(both 1 matches eval_compare_knee.py)",
        flush=True,
    )

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
        calib_pos_scores,
        label_matrix_cal,
        b_pos,
        b_neg,
        lambda_candidates,
        args.risk_metric,
        args.band_gamma_lower,
        args.band_gamma_upper,
    )

    control_idxs = resolve_control_label_indices(
        list(label_names),
        control_labels=getattr(args, "control_labels", None),
        exclude_labels=getattr(args, "exclude_labels", None),
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
        alpha_per_label=resolve_alpha_per_label(
            float(args.alpha),
            parse_alpha_per_label(args.alpha_per_label),
            int(label_matrix_cal.shape[1]),
        ),
        pvalue_method=str(getattr(args, "ltt_pvalue_method", "binomial")),
        control_both=bool(getattr(args, "control_fnr_and_fpr", False)),
        control_label_indices=control_idxs,
    )

    m_grid = int(lambda_candidates.shape[0])
    holm_tightest = float(args.delta) / max(m_grid, 1)
    print(
        f"dataset={dataset_tag} risk_metric={args.risk_metric} "
        f"LTT_conditional_test={args.ltt_conditional_test} "
        f"Grid m={m_grid}, delta={args.delta}. "
        f"LTT-Cond uses Holm; tightest per-step cutoff = delta/m = {holm_tightest:.3g}."
    )
    cond_fingerprint = hash(
        (
            args.alpha,
            args.delta,
            args.risk_metric,
            args.ltt_conditional_test,
            args.num_lambdas,
            args.seed,
            args.est_size,
            args.min_pos,
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
            f"  min r_hat={risks_arr[j_min_r]:.4f} at lambda={lambda_candidates[j_min_r]:.4f} "
            f"(n_used={int(ns_arr[j_min_r])}, p={p_arr[j_min_r]:.3g})"
        )
        print(
            f"  min p_value={p_arr[j_min_p]:.3g} at lambda={lambda_candidates[j_min_p]:.4f} "
            f"(r_hat={risks_arr[j_min_p]:.4f}, n_used={int(ns_arr[j_min_p])})"
        )
        if (
            key == "ltt_conditional"
            and sel.get("conditional_test") == "via_W"
            and sel.get("w_mean") is not None
        ):
            w_arr = np.asarray(sel["w_mean"], dtype=float)
            j_wm = int(np.argmin(w_arr))
            print(
                f"  via_W: mean W=S*(L-{float(args.alpha):g}) uses n_cal={losses_mat.shape[0]} for every λ;"
                f" Hoeffding p targets H0 E[W]>0 (equivalent to high R_cond when E[S]>0)."
            )
            print(
                f"  min W_mean={w_arr[j_wm]:.4f} at lambda={lambda_candidates[j_wm]:.4f} "
                f"(p={p_arr[j_wm]:.3g})"
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
            args.band_gamma_lower,
            args.band_gamma_upper,
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
    #     args.band_gamma_lower,
    #     args.band_gamma_upper,
    #     label_indices=control_idxs,
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

    conf_thr_curve, ent_thr_curve = fixed_threshold_schedule_curves(
        test_pos_scores,
        label_matrix_test,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
        label_indices=control_idxs,
        label_names=label_names,
    )
    conf_thr_legend = (
        confidence_baseline_legend_label(float(args.confidence_threshold))
        if conf_thr_curve is not None
        else None
    )
    ent_thr_legend = (
        entropy_baseline_legend_label(float(args.entropy_threshold))
        if ent_thr_curve is not None
        else None
    )

    file_tag = f"{dataset_tag}_{args.risk_metric}_asym"
    n_test = int(label_matrix_test.shape[0])
    alpha_k = resolve_alpha_per_label(
        float(args.alpha),
        parse_alpha_per_label(args.alpha_per_label),
        int(label_matrix_cal.shape[1]),
    )
    control_label_names = [label_names[i] for i in control_idxs]
    control_alpha_by_label = {label_names[i]: float(alpha_k[i]) for i in control_idxs}
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
        "variant": "eval_compare_knee_asymmetric_tau_band",
        "plot_conditional": bool(args.plot_conditional),
        "band_gamma_lower": float(args.band_gamma_lower),
        "band_gamma_upper": float(args.band_gamma_upper),
        "tau_grid_candidates": lambda_candidates.tolist(),
        "dataset": dataset_tag,
        "alpha": args.alpha,
        "delta": args.delta,
        "n_cal": int(label_matrix_cal.shape[0]),
        "n_est": int(label_matrix_est.shape[0]),
        "n_test": int(label_matrix_test.shape[0]),
        "risk_metric": args.risk_metric,
        "exclude_labels": getattr(args, "exclude_labels", None),
        "control_labels": getattr(args, "control_labels", None),
        "control_label_indices": control_idxs,
        "control_label_names": [label_names[i] for i in control_idxs],
        "abstain_penalty": float(args.abstain_penalty),
        "ltt_abstain_utility": args.ltt_abstain_utility,
        "ltt_conditional_utility": args.ltt_conditional_utility,
        "ltt_conditional_test": args.ltt_conditional_test,
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
