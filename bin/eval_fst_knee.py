#!/usr/bin/env python3
"""
FST evaluation for FastMRI knee slice/volume datasets (FNR control).

Implements LTT/FST over lambda for ambiguity-band stopping with:
  - all-sample averaging on calibration risk
  - abstain (never stopped by horizon) -> loss 0
  - fixed-sequence testing using Hoeffding p-values and FWER delta
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.data import MRDataModule
from classifier.masks.cumulative_masking import CumulativeMaskState
from classifier.utils.utils import get_config
from eval_conformal_fnr_fpr import (
    FORCED_PRED_THRESHOLD,
    build_mask_for_sampling_rate,
    compute_dual_thresholds,
    get_probs_labels,
    load_model,
)
from fst.fixed_sequence import fst_select_from_pvalues
from fst.conformal_triage import estimate_scalar_triage_endpoints
from fst.ltt import (
    compute_grid_pvalues,
    grid_risks_and_ns,
    grid_utilities,
    ltt_select_lambda,
)


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def split_dataset(dataset, calib_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n_calib = int(len(indices) * calib_fraction)
    calib_idx = indices[:n_calib]
    test_idx = indices[n_calib:]
    return Subset(dataset, calib_idx), Subset(dataset, test_idx)


def get_val_test_datasets(config, use_testset: bool):
    if config.dataset not in ["knee", "knee_vol"]:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    datamodule = MRDataModule(config=config, dev_mode=False)
    val_ds = datamodule.val_dataset
    test_ds = datamodule.test_dataset if use_testset else None
    return val_ds, test_ds


def build_prob_matrix(
    probs_by_label: Dict[str, np.ndarray], label_names: List[str], class_idx: int
) -> np.ndarray:
    return np.stack([probs_by_label[name][:, class_idx] for name in label_names], axis=1)


def build_label_matrix(
    labels_by_label: Dict[str, np.ndarray], label_names: List[str], positive_class_idx: int
) -> np.ndarray:
    return np.stack(
        [(labels_by_label[name].astype(int) == positive_class_idx).astype(int) for name in label_names],
        axis=1,
    )


def collect_labels(
    dataset,
    label_names: List[str],
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    loader = build_dataloader(dataset, batch_size, num_workers, shuffle=False)
    labels_by_label = {name: [] for name in label_names}
    for batch in loader:
        for name in label_names:
            labels_by_label[name].append(batch["label"][name].detach().cpu().squeeze(1).numpy())
    return {name: np.concatenate(vals, axis=0) for name, vals in labels_by_label.items()}


def split_estimation_set(
    dataset,
    label_matrix: np.ndarray,
    est_size: int,
    seed: int,
    min_pos: int,
) -> Tuple[Subset, Subset]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    est_count = min(est_size, len(indices))
    est_set = set(indices[:est_count].tolist())
    remaining_set = set(indices[est_count:].tolist())
    num_labels = label_matrix.shape[1]
    for label_idx in range(num_labels):
        pos_indices = np.where(label_matrix[:, label_idx] == 1)[0]
        while (
            pos_indices.size > 0
            and np.sum(label_matrix[list(est_set), label_idx] == 1) < min_pos
        ):
            pos_candidates = [i for i in pos_indices if i in remaining_set]
            if not pos_candidates:
                break
            pick = int(pos_candidates[0])
            est_set.add(pick)
            remaining_set.discard(pick)
    est_indices = np.array(sorted(est_set), dtype=int)
    remaining_indices = np.array([i for i in indices if i not in est_set], dtype=int)
    return Subset(dataset, est_indices), Subset(dataset, remaining_indices)


def find_optimal_delta(
    pos_scores: np.ndarray, neg_scores: np.ndarray, num_deltas: int
) -> Tuple[float, float, float]:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0, 1.0, 0.0
    deltas = np.linspace(0.0, 0.95, num_deltas)
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
    pos_scores: np.ndarray,
    label_matrix: np.ndarray,
    num_deltas: int,
    *,
    band_source: str = "quantile_sep",
    triage_desired_ppv: float = 0.9,
    triage_desired_npv: float = 0.95,
    triage_tolerance: float = 0.2,
    triage_num_thresh: int = 501,
    triage_min_high_bin: int = 10,
    triage_min_low_bin: int = 10,
    triage_fallback_quantile: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``b_pos``, ``b_neg`` per (timestep, label) on the estimation set.

    ``band_source``:
      - ``quantile_sep``: existing ``find_optimal_delta`` quantiles (default).
      - ``conformal_triage``: scalar thresholds from Angelopoulos et al. conformal triage
        (bins + binomial inversion); degenerate grids fall back to quantile_sep if
        ``triage_fallback_quantile``.
    """
    if band_source not in ("quantile_sep", "conformal_triage"):
        raise ValueError(f"unknown band_source: {band_source!r}")

    num_steps, num_labels = pos_scores.shape[1], pos_scores.shape[2]
    b_pos = np.zeros((num_steps, num_labels), dtype=float)
    b_neg = np.zeros((num_steps, num_labels), dtype=float)
    for t in range(num_steps):
        for label_idx in range(num_labels):
            scores_t = pos_scores[:, t, label_idx]
            labels_t = label_matrix[:, label_idx]
            pos_mask = labels_t == 1
            neg_mask = labels_t == 0

            if band_source == "quantile_sep":
                delta_star_t, b_pos_t, b_neg_t = find_optimal_delta(
                    scores_t[pos_mask], scores_t[neg_mask], num_deltas
                )
                _ = delta_star_t
                b_pos[t, label_idx] = b_pos_t if np.any(neg_mask) else 1.0
                b_neg[t, label_idx] = b_neg_t if np.any(pos_mask) else 0.0
                continue

            # conformal triage on scalar score (one row per acquisition step)
            use_quantile_fallback = False
            if not (np.any(pos_mask) and np.any(neg_mask)):
                use_quantile_fallback = True
            else:
                try:
                    lam_lo, lam_hi = estimate_scalar_triage_endpoints(
                        scores_t,
                        labels_t.astype(int),
                        desired_ppv=float(triage_desired_ppv),
                        desired_npv=float(triage_desired_npv),
                        tolerance=float(triage_tolerance),
                        num_thresh=int(triage_num_thresh),
                        min_high_bin=int(triage_min_high_bin),
                        min_low_bin=int(triage_min_low_bin),
                    )
                except Exception:
                    use_quantile_fallback = True
                else:
                    if lam_lo >= lam_hi or not (
                        np.isfinite(lam_lo) and np.isfinite(lam_hi)
                    ):
                        use_quantile_fallback = True
                    else:
                        b_neg[t, label_idx] = float(lam_lo)
                        b_pos[t, label_idx] = float(lam_hi)

            if use_quantile_fallback and triage_fallback_quantile:
                delta_star_t, b_pos_t, b_neg_t = find_optimal_delta(
                    scores_t[pos_mask], scores_t[neg_mask], num_deltas
                )
                _ = delta_star_t
                b_pos[t, label_idx] = b_pos_t if np.any(neg_mask) else 1.0
                b_neg[t, label_idx] = b_neg_t if np.any(pos_mask) else 0.0
            elif use_quantile_fallback:
                b_pos[t, label_idx] = 1.0 if not np.any(neg_mask) else float(
                    np.quantile(scores_t[neg_mask], 1.0)
                )
                b_neg[t, label_idx] = 0.0 if not np.any(pos_mask) else float(
                    np.quantile(scores_t[pos_mask], 0.0)
                )
    return b_pos, b_neg


def add_ambiguity_band_args(parser: argparse.ArgumentParser) -> None:
    """Shared CLI for ``compute_bands`` (quantile overlap vs conformal triage anchors)."""
    parser.add_argument(
        "--band_source",
        type=str,
        default="quantile_sep",
        choices=["quantile_sep", "conformal_triage"],
        help=(
            "How to set nominal overlap $[b^-,\, b^+]$ on the estimation set: quantile_sep (default) "
            "or conformal_triage (scalar thresholds from Angelopoulos et al. medRxiv 2024.02.09.24302543)."
        ),
    )
    parser.add_argument(
        "--triage_desired_ppv",
        type=float,
        default=0.8,
        help="Conformal triage target PPV (passed to selective high-threshold calibration).",
    )
    parser.add_argument(
        "--triage_desired_npv",
        type=float,
        default=0.5,
        help="Conformal triage target NPV (low-threshold selective group).",
    )
    parser.add_argument(
        "--triage_tolerance",
        type=float,
        default=0.2,
        help="Binomial UC level tolerance δ in conformal triage (repo default 0.2).",
    )
    parser.add_argument(
        "--triage_num_thresh",
        type=int,
        default=101,
        help="Grid size over [0,1] for triage threshold search.",
    )
    parser.add_argument(
        "--triage_min_high_bin",
        type=int,
        default=5,
        help="Minimum count in {score > λ} for high-side triage calibration (lower for tiny est sets).",
    )
    parser.add_argument(
        "--triage_min_low_bin",
        type=int,
        default=5,
        help="Minimum count in {score ≤ λ} for low-side triage calibration.",
    )
    parser.add_argument(
        "--no_triage_fallback_quantile",
        action="store_true",
        help=(
            "If conformal triage fails or yields b^- ≥ b^+, abstain from quantile fallback "
            "(use permissive/extreme clamps instead)."
        ),
    )


def compute_bands_from_argparse(
    pos_scores: np.ndarray,
    label_matrix: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    fb = not bool(getattr(args, "no_triage_fallback_quantile", False))
    return compute_bands(
        pos_scores,
        label_matrix,
        int(args.num_deltas),
        band_source=str(getattr(args, "band_source", "quantile_sep")),
        triage_desired_ppv=float(getattr(args, "triage_desired_ppv", 0.9)),
        triage_desired_npv=float(getattr(args, "triage_desired_npv", 0.95)),
        triage_tolerance=float(getattr(args, "triage_tolerance", 0.2)),
        triage_num_thresh=int(getattr(args, "triage_num_thresh", 501)),
        triage_min_high_bin=int(getattr(args, "triage_min_high_bin", 10)),
        triage_min_low_bin=int(getattr(args, "triage_min_low_bin", 10)),
        triage_fallback_quantile=fb,
    )


def triage_band_config_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "band_source": str(getattr(args, "band_source", "quantile_sep")),
        "triage_desired_ppv": float(getattr(args, "triage_desired_ppv", 0.9)),
        "triage_desired_npv": float(getattr(args, "triage_desired_npv", 0.95)),
        "triage_tolerance": float(getattr(args, "triage_tolerance", 0.2)),
        "triage_num_thresh": int(getattr(args, "triage_num_thresh", 501)),
        "triage_min_high_bin": int(getattr(args, "triage_min_high_bin", 10)),
        "triage_min_low_bin": int(getattr(args, "triage_min_low_bin", 10)),
        "triage_fallback_quantile": (not bool(getattr(args, "no_triage_fallback_quantile", False))),
    }


def print_ambiguity_band_diagnostics(
    sampling_rates: List[float],
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    label_names: List[str],
    num_deltas: int,
    *,
    band_source: str = "quantile_sep",
) -> None:
    """
    Print b_neg[t,k], b_pos[t,k] from compute_bands(est, …) — the ambiguity
    interval endpoints per sampling-rate row t and label k, **before** λ is
    applied in compute_bounds(λ, b_pos, b_neg).
    """
    t_steps, n_lab = int(b_pos.shape[0]), int(b_pos.shape[1])
    if len(label_names) < n_lab:
        label_names = list(label_names) + [f"lbl{j}" for j in range(len(label_names), n_lab)]
    if len(sampling_rates) != t_steps:
        print(
            "[ambiguity bands] WARN: len(sampling_rates)="
            f"{len(sampling_rates)} != b_pos.shape[0]={t_steps}; "
            "using index alignment.",
            flush=True,
        )
    print(
        f"[ambiguity bands] band_source={band_source!r}; estimated on est set only; "
        f"quantile scan uses num_deltas={num_deltas} when band_source=quantile_sep. "
        "Per (sampling_rate, label): ambiguity interval "
        "[b_neg, b_pos] in score units (positive-class prob). λ not applied yet.",
        flush=True,
    )
    names = label_names[:n_lab]
    for t in range(t_steps):
        sr = sampling_rates[t] if t < len(sampling_rates) else float(t)
        parts = []
        for k in range(n_lab):
            lo, hi = float(b_neg[t, k]), float(b_pos[t, k])
            inv = "" if hi > lo else "  (INVERTED: interval empty or flipped)"
            parts.append(f"  {names[k]}: b_neg={lo:.6f} b_pos={hi:.6f}{inv}")
        sr_str = f"{sr:.6g}"
        body = "\n".join(parts)
        print(f"  sampling_rate[{t}] = {sr_str}\n{body}", flush=True)
    print("", flush=True)


def compute_bounds(
    lam: float,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = lam * b_neg + (1.0 - lam) * 0.0
    upper = lam * b_pos + (1.0 - lam) * 1.0
    return lower, upper


def compute_stop_indices(
    pos_scores: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_samples, num_steps, _ = pos_scores.shape
    stop_indices = np.full(num_samples, num_steps - 1, dtype=int)
    stopped = np.zeros(num_samples, dtype=bool)
    for t in range(num_steps):
        if np.any(lower[t] > upper[t]):
            stop_mask = np.ones(num_samples, dtype=bool)
        else:
            decide = (pos_scores[:, t, :] < lower[t]) | (pos_scores[:, t, :] > upper[t])
            stop_mask = decide.all(axis=1)
        newly = np.logical_and(~stopped, stop_mask)
        stop_indices[newly] = t
        stopped[newly] = True
    return stop_indices, stopped


def compute_preds_at_stop(
    pos_scores: np.ndarray,
    stop_indices: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
) -> np.ndarray:
    n = pos_scores.shape[0]
    scores_at_stop = pos_scores[np.arange(n), stop_indices, :]
    b_pos_t = b_pos[stop_indices]
    b_neg_t = b_neg[stop_indices]
    center = b_neg_t / (1.0 + b_neg_t - b_pos_t)
    return (scores_at_stop >= center).astype(int)


def compute_sample_fnr_loss(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(preds == 1, labels == 1).sum(axis=1)
    pos_counts = labels.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = 1.0 - intersection / pos_counts
    return np.where(pos_counts > 0, loss, 0.0)


def compute_sample_fpr_loss(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    fp_counts = np.logical_and(preds == 1, labels == 0).sum(axis=1)
    neg_counts = labels.shape[1] - labels.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = fp_counts / neg_counts
    return np.where(neg_counts > 0, loss, 0.0)


def fnr_losses_with_abstain_zero(
    preds: np.ndarray,
    labels: np.ndarray,
    stopped: np.ndarray,
) -> np.ndarray:
    losses = np.zeros(labels.shape[0], dtype=float)
    if np.any(stopped):
        losses[stopped] = compute_sample_fnr_loss(preds[stopped], labels[stopped])
    return losses


def per_lambda_calibration_arrays_knee(
    pos_scores: np.ndarray,
    labels: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    lambda_candidates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (losses_mat, stopped_mat, stop_idx_mat) each (n, m).
    losses_mat[i, j] is per-sample FNR loss at the lambda_j stop time
    (0 for samples that did not stop).
    """
    n_cal = labels.shape[0]
    m = lambda_candidates.shape[0]
    losses_mat = np.zeros((n_cal, m), dtype=float)
    stopped_mat = np.zeros((n_cal, m), dtype=bool)
    stop_idx_mat = np.zeros((n_cal, m), dtype=int)
    for j, lam in enumerate(lambda_candidates):
        lower, upper = compute_bounds(float(lam), b_pos, b_neg)
        stop_indices, stopped = compute_stop_indices(pos_scores, lower, upper)
        preds = compute_preds_at_stop(pos_scores, stop_indices, b_pos, b_neg)
        losses_mat[:, j] = fnr_losses_with_abstain_zero(preds, labels, stopped)
        stopped_mat[:, j] = stopped
        stop_idx_mat[:, j] = stop_indices
    return losses_mat, stopped_mat, stop_idx_mat


def plot_results(
    output_dir: Path,
    sampling_rates: List[float],
    fst_stopped_count: List[int],
    casc_stopped_count: List[int],
    fnr_stats: Dict[str, Dict[str, List[float]]],
    fpr_stats: Dict[str, Dict[str, List[float]]],
    fnr_unconditional_cumulative: Dict[str, List[float]],
    fpr_unconditional_cumulative: Dict[str, List[float]],
    method_label: str = "FST",
    method_tag: str = "fst",
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.plot(sampling_rates, fst_stopped_count, marker="o", color="#4C72B0", label=method_label)
    ax.plot(sampling_rates, casc_stopped_count, marker="o", color="#55A868", label="CASC")
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Number of stopped samples")
    ax.set_title("Stopped samples vs sampling rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"{method_tag}_stopped_vs_sampling_rate.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
    colors = {method_tag: "#4C72B0", "casc": "#55A868", "forced": "#7F7F7F"}
    labels_map = {method_tag: method_label, "casc": "CASC", "forced": "Forced"}
    for name in [method_tag, "casc", "forced"]:
        ax.plot(
            sampling_rates,
            fnr_stats[name]["cumulative"],
            marker="o",
            color=colors[name],
            linestyle="-",
            label=f"{labels_map[name]} cumulative",
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Cumulative FNR")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cumulative FNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"{method_tag}_fnr_cumulative.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
    for name in [method_tag, "casc", "forced"]:
        ax.plot(
            sampling_rates,
            fpr_stats[name]["cumulative"],
            marker="o",
            color=colors[name],
            linestyle="-",
            label=f"{labels_map[name]} cumulative",
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Cumulative FPR")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cumulative FPR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / f"{method_tag}_fpr_cumulative.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
    for name in [method_tag, "casc", "forced"]:
        ax.plot(
            sampling_rates,
            fnr_unconditional_cumulative[name],
            marker="o",
            color=colors[name],
            linestyle="-",
            label=f"{labels_map[name]} unconditional cumulative",
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Unconditional cumulative FNR risk")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unconditional cumulative FNR risk")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"{method_tag}_fnr_unconditional_cumulative.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=300)
    for name in [method_tag, "casc", "forced"]:
        ax.plot(
            sampling_rates,
            fpr_unconditional_cumulative[name],
            marker="o",
            color=colors[name],
            linestyle="-",
            label=f"{labels_map[name]} unconditional cumulative",
        )
    ax.set_xlabel("Sampling rate")
    ax.set_ylabel("Unconditional cumulative FPR risk")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unconditional cumulative FPR risk")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"{method_tag}_fpr_unconditional_cumulative.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def confusion_from_binary_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Tuple[int, int, int, int]:
    tp = int(np.logical_and(preds == 1, labels == 1).sum())
    fp = int(np.logical_and(preds == 1, labels == 0).sum())
    tn = int(np.logical_and(preds == 0, labels == 0).sum())
    fn = int(np.logical_and(preds == 0, labels == 1).sum())
    return tp, fp, tn, fn


def fnr_fpr_from_confusion(tp: int, fp: int, tn: int, fn: int) -> Tuple[float, float]:
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(fnr), float(fpr)


def build_uniform_alpha_schedule(num_steps: int, alpha_total: float) -> np.ndarray:
    if num_steps <= 0:
        return np.array([], dtype=float)
    return np.full(num_steps, alpha_total / num_steps, dtype=float)


def build_score_cache_dir(output_dir: str, seed: int) -> Path:
    return Path(output_dir) / "score_cache" / f"seed_{seed}"


def load_cached_scores(cache_path: Path) -> Optional[Dict]:
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    return None


def save_cached_scores(cache_path: Path, payload: Dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def main():
    parser = argparse.ArgumentParser(description="FST evaluation for knee/knee_vol.")
    parser.add_argument("--config_path", type=str, default="configs/kspace/knee_slice_bayes_distill.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="/gpfs/data/chopralab/us453/kspace_classifier/checkpoints/knee_slice_bayes_distill/2xtvl5na/checkpoints/bayes_distill_arms-knee-resnet18-epoch=16-best_metric=0.91.ckpt")
    parser.add_argument("--alpha", type=float, default=0.1, help="Target FNR risk level.")
    parser.add_argument("--delta", type=float, default=0.1, help="FWER level for FST.")
    parser.add_argument("--num_deltas", type=int, default=51)
    parser.add_argument("--est_size", type=int, default=50)
    parser.add_argument("--min_pos", type=int, default=5)
    parser.add_argument("--positive_class_idx", type=int, default=1)
    parser.add_argument("--calib_fraction", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output_dir", type=str, default="./fst_results/knee_slice")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling_rates", type=str, default="0.05,0.08,0.125,0.15,0.25")
    parser.add_argument("--use_testset", action="store_true")
    parser.add_argument("--num_lambdas", type=int, default=201)
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
        "--selection_method",
        type=str,
        default="fst",
        choices=["fst", "ltt"],
        help="FST (fixed sequence) or LTT (Holm step-down + utility selection).",
    )
    parser.add_argument(
        "--risk_mode",
        type=str,
        default="unconditional_with_abstain",
        choices=["unconditional_with_abstain", "conditional_on_stopped"],
        help=(
            "How to define R(lambda): mean over all calibration samples with "
            "abstainers contributing --abstain_penalty, or mean over stopped only."
        ),
    )
    parser.add_argument(
        "--abstain_penalty",
        type=float,
        default=0.0,
        help="Loss assigned to non-stopped (abstained) calibration samples in unconditional mode.",
    )
    parser.add_argument(
        "--utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk"],
        help="Utility function used by LTT to pick a lambda from the safe set.",
    )
    parser.add_argument(
        "--default_lambda_idx",
        type=int,
        default=None,
        help="If LTT safe set is empty, fall back to lambda_candidates[default_lambda_idx]. If unset, returns None.",
    )
    parser.add_argument(
        "--print_ambiguity_bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print b_neg/b_pos from compute_bands per sampling-rate and label "
            "(before λ is applied via compute_bounds). Pass --no-print_ambiguity_bands to silence."
        ),
    )
    add_ambiguity_band_args(parser)
    args = parser.parse_args()

    config = get_config(args.config_path)
    device = torch.device(args.device)
    label_names = config.label_names.split(",") if isinstance(config.label_names, str) else config.label_names
    input_type = getattr(config, "input_type", "kspace")
    is_volume = "vol" in str(getattr(config, "dataset", "")).lower()

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

    est_loader = build_dataloader(est_ds, batch_size, args.num_workers, shuffle=False)
    calib_loader = build_dataloader(calib_ds, batch_size, args.num_workers, shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size, args.num_workers, shuffle=False)

    est_mask_state = CumulativeMaskState(len(est_ds)) if args.cumulative_sampling else None
    calib_mask_state = CumulativeMaskState(len(calib_ds)) if args.cumulative_sampling else None
    test_mask_state = CumulativeMaskState(len(test_ds)) if args.cumulative_sampling else None
    score_cache_dir = build_score_cache_dir(args.output_dir, args.seed)

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
                cumulative_state=est_mask_state, input_type=input_type, is_volume=is_volume
            )
            cal_probs, cal_labels = get_probs_labels(
                model, calib_loader, label_names, device, mask_fn,
                cumulative_state=calib_mask_state, input_type=input_type, is_volume=is_volume
            )
            tst_probs, tst_labels = get_probs_labels(
                model, test_loader, label_names, device, mask_fn,
                cumulative_state=test_mask_state, input_type=input_type, is_volume=is_volume
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

    b_pos, b_neg = compute_bands_from_argparse(
        est_pos_scores, label_matrix_est, args
    )
    if args.print_ambiguity_bands:
        print_ambiguity_band_diagnostics(
            sampling_rates, b_pos, b_neg, label_names, args.num_deltas,
            band_source=str(args.band_source),
        )
    losses_mat, stopped_mat, stop_idx_mat = per_lambda_calibration_arrays_knee(
        calib_pos_scores, label_matrix_cal, b_pos, b_neg, lambda_candidates
    )
    risks, ns = grid_risks_and_ns(
        losses_mat, stopped_mat, args.risk_mode, args.abstain_penalty
    )
    p_values = compute_grid_pvalues(risks, ns, args.alpha, B=1.0)
    utilities = grid_utilities(stopped_mat, stop_idx_mat, risks, args.utility)

    if args.selection_method == "fst":
        lambda_hat, meta = fst_select_from_pvalues(lambda_candidates, p_values, args.delta)
        meta_key = "fst_meta"
    else:
        lambda_hat, meta = ltt_select_lambda(
            lambda_candidates,
            risks,
            p_values,
            args.delta,
            utilities,
            default_lambda_idx=args.default_lambda_idx,
        )
        meta_key = "ltt_meta"

    print(f"selection_method={args.selection_method} risk_mode={args.risk_mode}")
    for j in range(lambda_candidates.shape[0]):
        print(
            f"j={j:4d} lambda={float(lambda_candidates[j]):.6f} "
            f"r_hat={float(risks[j]):.6f} n={int(ns[j])} "
            f"p={float(p_values[j]):.6g} u={float(utilities[j]):.4f} "
            f"stopped={int(stopped_mat[:, j].sum())}"
        )
    print(f"{args.selection_method} reason: {meta['reason']}")
    print(f"Selected lambda_hat: {lambda_hat}")

    out = {
        "alpha": args.alpha,
        "delta": args.delta,
        "selection_method": args.selection_method,
        "risk_mode": args.risk_mode,
        "abstain_penalty": float(args.abstain_penalty),
        "utility": args.utility,
        "default_lambda_idx": args.default_lambda_idx,
        "lambda_hat": lambda_hat,
        meta_key: meta,
        "ambiguity_band": triage_band_config_dict(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if lambda_hat is not None:
        method_tag = args.selection_method
        method_label = args.selection_method.upper()
        lower, upper = compute_bounds(float(lambda_hat), b_pos, b_neg)
        stop_indices, stopped = compute_stop_indices(test_pos_scores, lower, upper)
        preds = compute_preds_at_stop(test_pos_scores, stop_indices, b_pos, b_neg)
        losses = fnr_losses_with_abstain_zero(preds, label_matrix_test, stopped)
        out["test_mean_risk"] = float(np.mean(losses))
        n_test, t_steps, n_labels = test_pos_scores.shape
        method_stopped_count, casc_stopped_count = [], []

        fnr_stats = {
            method_tag: {"stopped": [], "cumulative": []},
            "casc": {"stopped": [], "cumulative": []},
            "forced": {"stopped": [], "cumulative": []},
        }
        fpr_stats = {
            method_tag: {"stopped": [], "cumulative": []},
            "casc": {"stopped": [], "cumulative": []},
            "forced": {"stopped": [], "cumulative": []},
        }
        fnr_unconditional_cumulative = {method_tag: [], "casc": [], "forced": []}
        fpr_unconditional_cumulative = {method_tag: [], "casc": [], "forced": []}

        # Selected-method per-rate stopped/cumulative metrics.
        for t in range(t_steps):
            idx_stopped = np.where(stopped & (stop_indices == t))[0]
            idx_cumulative = np.where(stopped & (stop_indices <= t))[0]
            method_stopped_count.append(int(idx_cumulative.size))
            if idx_stopped.size == 0:
                fnr_s, fpr_s = 0.0, 0.0
            else:
                tp, fp, tn, fn = confusion_from_binary_predictions(
                    preds[idx_stopped], label_matrix_test[idx_stopped]
                )
                fnr_s, fpr_s = fnr_fpr_from_confusion(tp, fp, tn, fn)
            if idx_cumulative.size == 0:
                fnr_c, fpr_c = 0.0, 0.0
            else:
                tp, fp, tn, fn = confusion_from_binary_predictions(
                    preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
                fnr_c, fpr_c = fnr_fpr_from_confusion(tp, fp, tn, fn)
            fnr_stats[method_tag]["stopped"].append(fnr_s)
            fnr_stats[method_tag]["cumulative"].append(fnr_c)
            fpr_stats[method_tag]["stopped"].append(fpr_s)
            fpr_stats[method_tag]["cumulative"].append(fpr_c)
            losses_fnr = np.zeros(n_test, dtype=float)
            losses_fpr = np.zeros(n_test, dtype=float)
            if idx_cumulative.size > 0:
                losses_fnr[idx_cumulative] = compute_sample_fnr_loss(
                    preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
                losses_fpr[idx_cumulative] = compute_sample_fpr_loss(
                    preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
            fnr_unconditional_cumulative[method_tag].append(float(np.mean(losses_fnr)))
            fpr_unconditional_cumulative[method_tag].append(float(np.mean(losses_fpr)))

        # CASC per-rate thresholds and cumulative stopping.
        alpha_k = build_uniform_alpha_schedule(t_steps, args.alpha)
        casc_active = np.ones(n_test, dtype=bool)
        casc_stop_indices = np.full(n_test, -1, dtype=int)
        casc_preds = np.zeros((n_test, n_labels), dtype=int)
        for t in range(t_steps):
            pred_sets_by_label: List[List[List[int]]] = []
            for label_idx in range(n_labels):
                cal_probs_lbl = np.stack(
                    [1.0 - calib_pos_scores[:, t, label_idx], calib_pos_scores[:, t, label_idx]], axis=1
                )
                cal_labels_lbl = label_matrix_cal[:, label_idx].astype(int)
                thr = compute_dual_thresholds(
                    cal_probs_lbl,
                    cal_labels_lbl,
                    alpha_fnr=float(alpha_k[t]),
                    alpha_fpr=float(alpha_k[t]),
                    pos_idx=args.positive_class_idx,
                )
                test_probs_lbl = np.stack(
                    [1.0 - test_pos_scores[:, t, label_idx], test_pos_scores[:, t, label_idx]], axis=1
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
                    pred_vec[label_idx] = 1 if s[0] == args.positive_class_idx else 0
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
            casc_stopped_count.append(int(idx_cumulative.size))
            if idx_stopped.size == 0:
                fnr_s, fpr_s = 0.0, 0.0
            else:
                tp, fp, tn, fn = confusion_from_binary_predictions(
                    casc_preds[idx_stopped], label_matrix_test[idx_stopped]
                )
                fnr_s, fpr_s = fnr_fpr_from_confusion(tp, fp, tn, fn)
            if idx_cumulative.size == 0:
                fnr_c, fpr_c = 0.0, 0.0
            else:
                tp, fp, tn, fn = confusion_from_binary_predictions(
                    casc_preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
                fnr_c, fpr_c = fnr_fpr_from_confusion(tp, fp, tn, fn)
            fnr_stats["casc"]["stopped"].append(fnr_s)
            fnr_stats["casc"]["cumulative"].append(fnr_c)
            fpr_stats["casc"]["stopped"].append(fpr_s)
            fpr_stats["casc"]["cumulative"].append(fpr_c)
            losses_fnr = np.zeros(n_test, dtype=float)
            losses_fpr = np.zeros(n_test, dtype=float)
            if idx_cumulative.size > 0:
                losses_fnr[idx_cumulative] = compute_sample_fnr_loss(
                    casc_preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
                losses_fpr[idx_cumulative] = compute_sample_fpr_loss(
                    casc_preds[idx_cumulative], label_matrix_test[idx_cumulative]
                )
            fnr_unconditional_cumulative["casc"].append(float(np.mean(losses_fnr)))
            fpr_unconditional_cumulative["casc"].append(float(np.mean(losses_fpr)))

        # Forced metrics by rate (stopped == cumulative since all are forced each rate).
        for t in range(t_steps):
            forced_preds = (test_pos_scores[:, t, :] >= FORCED_PRED_THRESHOLD).astype(int)
            tp, fp, tn, fn = confusion_from_binary_predictions(forced_preds, label_matrix_test)
            fnr_t, fpr_t = fnr_fpr_from_confusion(tp, fp, tn, fn)
            fnr_stats["forced"]["stopped"].append(fnr_t)
            fnr_stats["forced"]["cumulative"].append(fnr_t)
            fpr_stats["forced"]["stopped"].append(fpr_t)
            fpr_stats["forced"]["cumulative"].append(fpr_t)
            forced_losses_fnr = compute_sample_fnr_loss(forced_preds, label_matrix_test)
            forced_losses_fpr = compute_sample_fpr_loss(forced_preds, label_matrix_test)
            fnr_unconditional_cumulative["forced"].append(float(np.mean(forced_losses_fnr)))
            fpr_unconditional_cumulative["forced"].append(float(np.mean(forced_losses_fpr)))

        plot_results(
            output_dir,
            sampling_rates,
            method_stopped_count,
            casc_stopped_count,
            fnr_stats,
            fpr_stats,
            fnr_unconditional_cumulative,
            fpr_unconditional_cumulative,
            method_label=method_label,
            method_tag=method_tag,
        )
        out["curves"] = {
            "sampling_rates": sampling_rates,
            f"{method_tag}_stopped_count": method_stopped_count,
            "casc_stopped_count": casc_stopped_count,
            "fnr": fnr_stats,
            "fpr": fpr_stats,
            "fnr_unconditional_cumulative": fnr_unconditional_cumulative,
            "fpr_unconditional_cumulative": fpr_unconditional_cumulative,
        }

    out_path = output_dir / f"{args.selection_method}_knee_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved summary -> {out_path}")


if __name__ == "__main__":
    main()

