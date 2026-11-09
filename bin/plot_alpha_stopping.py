#!/usr/bin/env python3
"""
RCS / LTT-conditional stopping behavior across control levels α.

Default (publication) output for FastMRI knee + Prostate PI-RADS
(no violins, no panel letters):

  * ``alpha_stopping_mri_panel`` — 2×N composite
      top:    stop fraction vs sampling rate (one line per α; one panel per dataset)
      bottom: expected acquisition E[r_stop] vs α (one panel per dataset)
  * ``alpha_stopping_mri_panel_top`` — stop-fraction row only

Legacy separate figures (incl. violins) via ``--also_split_figures``.

Reuses score caches / prostate prediction CSVs (or a saved prostate ``.npz`` cache)
and only re-runs λ̂ selection over the α grid.

Examples
--------
Knee + Prostate (preferred)::

    python bin/plot_alpha_stopping.py \\
      --datasets knee_slice,prostate_pirads \\
      --prostate_score_cache data_cache/prostate_path15_scores.npz \\
      --out_dir ./plots/alpha_stopping_mri

Replot from a saved JSON (no recompute)::

    python bin/plot_alpha_stopping.py --from_json plots/alpha_stopping_mri/alpha_stopping_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_figure_export import (
    COMPARE_LEGEND_FONTSIZE,
    COMPARE_PUBLICATION_DPI,
    COMPARE_XLABEL_SAMPLING,
    configure_compare_plot_style,
    save_compare_figure,
    style_compare_ax,
)
from eval_compare_knee import (
    parse_alpha_per_label,
    per_lambda_calibration_arrays_knee_metric,
    resolve_alpha_per_label,
    resolve_control_label_indices,
    select_lambda_methods,
)
from eval_fst_knee import (
    build_label_matrix,
    build_prob_matrix,
    compute_bands,
    compute_bounds,
    compute_stop_indices,
    load_cached_scores,
)

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent

DEFAULT_PATHS = {
    "knee_slice": {
        "summary": WORKSPACE / "new_plots/knee_slice_fnr/testset/compare_knee_summary_knee_slice_fnr.json",
        "score_cache": WORKSPACE / "new_plots/knee_slice_fnr/testset/score_cache/seed_0",
    },
    "cifar10": {
        "summary": WORKSPACE / "neurips_plots/cifar10_compare/testset/compare_cifar10_summary_max_score.json",
        "score_cache": WORKSPACE / "neurips_plots/cifar10_compare/testset/score_cache/seed_42",
    },
    "imagenet_hf": {
        "summary": (
            WORKSPACE
            / "neurips_plots/imagenet_hierarchical_compare/testset/dogs_german_shepherd"
            / "compare_imagenet_hierarchical_summary_imagenet_hf_dogs_german_shepherd_fnr.json"
        ),
        "score_cache": (
            WORKSPACE
            / "neurips_plots/imagenet_hierarchical_compare/testset/dogs_german_shepherd"
            / "score_cache/seed_42"
        ),
    },
    "prostate_pirads": {
        "summary": (
            WORKSPACE / "neurips_plots/prostate_pirads_fnr/fnr/compare_prostate_pirads_summary_prostate_pirads_fnr.json"
        ),
    },
}

DATASET_DISPLAY = {
    "knee_slice": "FastMRI knee",
    "prostate_pirads": "Prostate PI-RADS",
    "cifar10": "CIFAR-10",
    "imagenet_hf": "ImageNet (GS)",
}

# Discrete, print-friendly α colors (ColorBrewer-like; avoid yellow-on-white).
ALPHA_PALETTE = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#CCB974",  # sand
    "#64B5CD",  # cyan
]
DATASET_COLORS = {
    "knee_slice": "#2F4B7C",
    "prostate_pirads": "#A51C30",
    "cifar10": "#3C5488",
    "imagenet_hf": "#E18727",
}


@dataclass
class AlphaSweepResult:
    dataset: str
    display_name: str
    sampling_rates: List[float]
    alphas: List[float]
    delta: float
    n_test: int
    lambda_hat: List[Optional[float]]
    stop_frac: List[List[float]]  # [α][t]
    mean_r_stop: List[float]
    stop_frac_final: List[float]
    # For violin: selected α -> per-sample stop rates
    stop_rates_by_alpha: Dict[str, List[float]] = field(default_factory=dict)
    stopped_mask_by_alpha: Dict[str, List[bool]] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float list: {raw!r}")
    return vals


def _band_kwargs_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    band = dict(summary.get("ambiguity_band") or {})
    return {
        "band_source": str(band.get("band_source", "quantile_sep")),
        "triage_desired_ppv": float(band.get("triage_desired_ppv", 0.8)),
        "triage_desired_npv": float(band.get("triage_desired_npv", 0.5)),
        "triage_tolerance": float(band.get("triage_tolerance", 0.2)),
        "triage_num_thresh": int(band.get("triage_num_thresh", 101)),
        "triage_min_high_bin": int(band.get("triage_min_high_bin", 5)),
        "triage_min_low_bin": int(band.get("triage_min_low_bin", 5)),
        "triage_fallback_quantile": bool(band.get("triage_fallback_quantile", True)),
    }


def _compute_bands(
    est_pos: np.ndarray,
    est_y: np.ndarray,
    summary: Dict[str, Any],
    num_deltas: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    kw = _band_kwargs_from_summary(summary)
    return compute_bands(est_pos, est_y, num_deltas, **kw)


def _stop_curve_from_indices(
    stop_idx: np.ndarray,
    stopped: np.ndarray,
    sampling_rates: Sequence[float],
) -> Tuple[List[float], float, float]:
    """Cumulative stop fraction + E[r_stop] using stop_idx (T-1 fill for never-stopped)."""
    rates = np.asarray(sampling_rates, dtype=float)
    t_steps = len(rates)
    n = int(stop_idx.shape[0])
    stop_frac = []
    for t in range(t_steps):
        # Samples that have stopped by timestep t (cleared ambiguity).
        stop_frac.append(float(np.mean(stopped & (stop_idx <= t))))
    # Charge last grid rate for never-stopped (matches compute_stop_indices fill).
    mean_r = float(np.mean(rates[np.clip(stop_idx, 0, t_steps - 1)]))
    final_frac = float(np.mean(stopped))
    return stop_frac, mean_r, final_frac


def _rcs_select_and_eval(
    *,
    dataset: str,
    display_name: str,
    sampling_rates: List[float],
    est_pos: np.ndarray,
    cal_pos: np.ndarray,
    test_pos: np.ndarray,
    est_y: np.ndarray,
    cal_y: np.ndarray,
    test_y: np.ndarray,
    summary: Dict[str, Any],
    alphas: Sequence[float],
    violin_alphas: Sequence[float],
    num_lambdas: int,
    delta: float,
    utility: str,
    control_idxs: Optional[List[int]],
    alpha_policy: str,
    alpha_per_label_ref: Optional[np.ndarray],
) -> AlphaSweepResult:
    b_pos, b_neg = _compute_bands(est_pos, est_y, summary)
    lambdas = np.linspace(0.0, 1.0, int(num_lambdas))
    risk_metric = str(summary.get("risk_metric", "fnr"))
    losses_mat, stopped_mat, stop_idx_mat, preds_cube = per_lambda_calibration_arrays_knee_metric(
        cal_pos, cal_y, b_pos, b_neg, lambdas, risk_metric
    )

    k_labels = int(cal_y.shape[1])
    ref_alpha = float(summary.get("alpha", alphas[0]))
    if alpha_per_label_ref is None and summary.get("alpha_per_label") is not None:
        alpha_per_label_ref = np.asarray(summary["alpha_per_label"], dtype=float)

    lambda_hats: List[Optional[float]] = []
    stop_fracs: List[List[float]] = []
    mean_rs: List[float] = []
    finals: List[float] = []
    stop_rates_by_alpha: Dict[str, List[float]] = {}
    stopped_mask_by_alpha: Dict[str, List[bool]] = {}
    violin_set = {float(a) for a in violin_alphas}

    for alpha in alphas:
        if alpha_policy == "scale" and alpha_per_label_ref is not None:
            scale = float(alpha) / max(ref_alpha, 1e-12)
            alpha_k = np.clip(np.asarray(alpha_per_label_ref, dtype=float) * scale, 0.0, 1.0)
        else:
            alpha_k = resolve_alpha_per_label(float(alpha), None, k_labels)

        sel = select_lambda_methods(
            lambdas,
            losses_mat,
            stopped_mat,
            stop_idx_mat,
            float(alpha),
            float(delta),
            float(summary.get("abstain_penalty", 0.03)),
            str(summary.get("ltt_abstain_utility", "stopped_count")),
            utility,
            None,
            None,
            preds_cube=preds_cube,
            labels=cal_y,
            risk_metric=risk_metric,
            alpha_per_label=alpha_k,
            pvalue_method=str(summary.get("ltt_pvalue_method", "binomial")),
            control_both=bool(summary.get("control_fnr_and_fpr", False)),
            control_label_indices=control_idxs,
        )
        lam = sel["ltt_conditional"]["lambda_hat"]
        reason = sel["ltt_conditional"]["meta"].get("reason")
        print(
            f"  [{dataset}] α={float(alpha):g} → λ̂="
            f"{'None' if lam is None else f'{float(lam):.4f}'} ({reason})",
            flush=True,
        )
        lambda_hats.append(None if lam is None else float(lam))

        if lam is None:
            zeros = [0.0] * len(sampling_rates)
            stop_fracs.append(zeros)
            # Never stop → charge full last rate on the grid.
            mean_rs.append(float(sampling_rates[-1]))
            finals.append(0.0)
            if float(alpha) in violin_set or any(
                abs(float(alpha) - a) < 1e-12 for a in violin_set
            ):
                key = f"{float(alpha):g}"
                stop_rates_by_alpha[key] = [float(sampling_rates[-1])] * int(test_pos.shape[0])
                stopped_mask_by_alpha[key] = [False] * int(test_pos.shape[0])
            continue

        lower, upper = compute_bounds(float(lam), b_pos, b_neg)
        stop_idx, stopped = compute_stop_indices(test_pos, lower, upper)
        frac, mean_r, final = _stop_curve_from_indices(stop_idx, stopped, sampling_rates)
        stop_fracs.append(frac)
        mean_rs.append(mean_r)
        finals.append(final)

        if any(abs(float(alpha) - a) < 1e-12 for a in violin_set):
            key = f"{float(alpha):g}"
            rates = np.asarray(sampling_rates, dtype=float)
            stop_rates_by_alpha[key] = rates[stop_idx].astype(float).tolist()
            stopped_mask_by_alpha[key] = stopped.astype(bool).tolist()

    return AlphaSweepResult(
        dataset=dataset,
        display_name=display_name,
        sampling_rates=[float(r) for r in sampling_rates],
        alphas=[float(a) for a in alphas],
        delta=float(delta),
        n_test=int(test_pos.shape[0]),
        lambda_hat=lambda_hats,
        stop_frac=stop_fracs,
        mean_r_stop=mean_rs,
        stop_frac_final=finals,
        stop_rates_by_alpha=stop_rates_by_alpha,
        stopped_mask_by_alpha=stopped_mask_by_alpha,
        notes={
            "alpha_policy": alpha_policy,
            "utility": utility,
            "num_lambdas": int(num_lambdas),
            "control_idxs": control_idxs,
        },
    )


def load_knee_like_sr_cache(
    score_cache_dir: Path,
    sampling_rates: Sequence[float],
    *,
    positive_class_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    est_list, cal_list, test_list = [], [], []
    label_names: Optional[List[str]] = None
    est_y = cal_y = test_y = None
    for rate in sampling_rates:
        path = score_cache_dir / f"sr_{float(rate):.6f}.pt"
        cached = load_cached_scores(path)
        if cached is None:
            raise FileNotFoundError(f"Missing score cache: {path}")
        names = list(cached["label_names"])
        if label_names is None:
            label_names = names
            est_y = build_label_matrix(cached["est_labels"], names, positive_class_idx)
            cal_y = build_label_matrix(cached["cal_labels"], names, positive_class_idx)
            test_y = build_label_matrix(cached["test_labels"], names, positive_class_idx)
        est_list.append(build_prob_matrix(cached["est_probs"], names, positive_class_idx))
        cal_list.append(build_prob_matrix(cached["cal_probs"], names, positive_class_idx))
        test_list.append(build_prob_matrix(cached["test_probs"], names, positive_class_idx))
    assert label_names is not None and est_y is not None and cal_y is not None and test_y is not None
    return (
        np.stack(est_list, axis=1),
        np.stack(cal_list, axis=1),
        np.stack(test_list, axis=1),
        est_y,
        cal_y,
        test_y,
        label_names,
    )


def sweep_knee_slice(
    summary_path: Path,
    score_cache: Path,
    alphas: Sequence[float],
    violin_alphas: Sequence[float],
    num_lambdas: int,
    alpha_policy: str,
) -> AlphaSweepResult:
    summary = json.loads(summary_path.read_text())
    rates = [float(r) for r in summary["curves"]["sampling_rates"]]
    est, cal, test, est_y, cal_y, test_y, label_names = load_knee_like_sr_cache(
        score_cache, rates, positive_class_idx=1
    )
    control_idxs = resolve_control_label_indices(
        label_names,
        control_labels=summary.get("control_labels"),
        exclude_labels=summary.get("exclude_labels"),
    )
    return _rcs_select_and_eval(
        dataset="knee_slice",
        display_name=DATASET_DISPLAY["knee_slice"],
        sampling_rates=rates,
        est_pos=est,
        cal_pos=cal,
        test_pos=test,
        est_y=est_y,
        cal_y=cal_y,
        test_y=test_y,
        summary=summary,
        alphas=alphas,
        violin_alphas=violin_alphas,
        num_lambdas=num_lambdas,
        delta=float(summary.get("delta", 0.1)),
        utility=str(summary.get("ltt_conditional_utility", "stopped_count")),
        control_idxs=control_idxs,
        alpha_policy=alpha_policy,
        alpha_per_label_ref=parse_alpha_per_label(
            ",".join(str(x) for x in summary["alpha_per_label"])
            if summary.get("alpha_per_label") is not None
            else None
        ),
    )


def sweep_imagenet(
    summary_path: Path,
    score_cache: Path,
    alphas: Sequence[float],
    violin_alphas: Sequence[float],
    num_lambdas: int,
    alpha_policy: str,
) -> AlphaSweepResult:
    summary = json.loads(summary_path.read_text())
    rates = [float(r) for r in summary["curves"]["sampling_rates"]]
    est, cal, test, est_y, cal_y, test_y, label_names = load_knee_like_sr_cache(
        score_cache, rates, positive_class_idx=int(summary.get("positive_class_idx", 1))
    )
    return _rcs_select_and_eval(
        dataset="imagenet_hf",
        display_name=DATASET_DISPLAY["imagenet_hf"],
        sampling_rates=rates,
        est_pos=est,
        cal_pos=cal,
        test_pos=test,
        est_y=est_y,
        cal_y=cal_y,
        test_y=test_y,
        summary=summary,
        alphas=alphas,
        violin_alphas=violin_alphas,
        num_lambdas=num_lambdas,
        delta=float(summary.get("delta", 0.075)),
        utility=str(summary.get("ltt_conditional_utility", "stopped_count")),
        control_idxs=list(range(int(cal_y.shape[1]))),
        alpha_policy="broadcast",
        alpha_per_label_ref=None,
    )


def sweep_prostate(
    summary_path: Path,
    analysis_dir: Optional[Path],
    alphas: Sequence[float],
    violin_alphas: Sequence[float],
    num_lambdas: int,
    alpha_policy: str,
    seed: int,
    est_size: int,
    *,
    score_cache: Optional[Path] = None,
    val_csv: Optional[Path] = None,
    test_csv: Optional[Path] = None,
) -> AlphaSweepResult:
    from eval_compare_prostate_pirads import (
        build_pirads_score_matrix,
        load_prediction_columns,
        pirads_sampling_rates,
        resolve_repo_path,
        split_val_into_est_calib,
    )

    summary = json.loads(summary_path.read_text())

    if score_cache is not None and score_cache.is_file():
        payload = np.load(score_cache, allow_pickle=True)
        rates = [float(r) for r in np.asarray(payload["sampling_rates"]).tolist()]
        est_pos = np.asarray(payload["est_pos"], dtype=np.float64)
        cal_pos = np.asarray(payload["cal_pos"], dtype=np.float64)
        test_pos = np.asarray(payload["test_pos"], dtype=np.float64)
        est_y = np.asarray(payload["est_y"], dtype=int)
        cal_y = np.asarray(payload["cal_y"], dtype=int)
        test_y = np.asarray(payload["test_y"], dtype=int)
        print(f"  [prostate] loaded score cache {score_cache}", flush=True)
    else:
        if analysis_dir is None and (val_csv is None or test_csv is None):
            raise FileNotFoundError(
                "Prostate requires an existing --prostate_score_cache .npz, "
                "or --prostate_analysis_dir / --prostate_val_csv+--prostate_test_csv. "
                "The Jul path15 analysis CSVs are no longer on GPFS."
            )
        averaging_tags = None
        if analysis_dir is not None:
            analysis_summary_path = analysis_dir / "summary.json"
            if not analysis_summary_path.is_file():
                raise FileNotFoundError(f"Missing analysis summary: {analysis_summary_path}")
            analysis = json.loads(analysis_summary_path.read_text())
            averaging_tags = np.asarray(analysis["averaging_tags"])
            outputs = analysis.get("outputs", {})
            if val_csv is None:
                raw = outputs.get("val_ensemble_predictions") or analysis.get("val_predictions_csv")
                val_csv = resolve_repo_path(str(raw)) if raw else None
            if test_csv is None:
                raw = outputs.get("test_ensemble_predictions") or analysis.get("test_predictions_csv")
                test_csv = resolve_repo_path(str(raw)) if raw else None
            if val_csv is None or not Path(val_csv).is_file():
                cand = list(analysis_dir.glob("*val*ensemble*.csv"))
                if cand:
                    val_csv = cand[0]
            if test_csv is None or not Path(test_csv).is_file():
                cand = list(analysis_dir.glob("*test*ensemble*.csv"))
                if cand:
                    test_csv = cand[0]
        if averaging_tags is None:
            averaging_tags = np.asarray(summary.get("averaging_tags"), dtype=str)

        val_csv_p = Path(val_csv) if val_csv is not None else None
        test_csv_p = Path(test_csv) if test_csv is not None else None
        if val_csv_p is None or test_csv_p is None or not val_csv_p.is_file() or not test_csv_p.is_file():
            raise FileNotFoundError(
                f"Prostate prediction CSVs not found (val={val_csv_p}, test={test_csv_p})."
            )

        rates, _ = pirads_sampling_rates(averaging_tags)
        val_scores, val_labels, _ = build_pirads_score_matrix(
            load_prediction_columns(val_csv_p), averaging_tags
        )
        test_scores, test_labels, _ = build_pirads_score_matrix(
            load_prediction_columns(test_csv_p), averaging_tags
        )
        est_idx, cal_idx = split_val_into_est_calib(
            val_scores.shape[0], est_size, 1.0, seed, True
        )
        est_pos = val_scores[est_idx][:, :, None]
        cal_pos = val_scores[cal_idx][:, :, None]
        test_pos = test_scores[:, :, None]
        est_y = val_labels[est_idx][:, None].astype(int)
        cal_y = val_labels[cal_idx][:, None].astype(int)
        test_y = test_labels[:, None].astype(int)

        if score_cache is not None:
            score_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                score_cache,
                sampling_rates=np.asarray(rates, dtype=float),
                averaging_tags=np.asarray(averaging_tags),
                est_pos=est_pos,
                cal_pos=cal_pos,
                test_pos=test_pos,
                est_y=est_y,
                cal_y=cal_y,
                test_y=test_y,
                seed=int(seed),
                est_size=int(est_size),
            )
            print(f"  [prostate] wrote score cache {score_cache}", flush=True)

    return _rcs_select_and_eval(
        dataset="prostate_pirads",
        display_name=DATASET_DISPLAY["prostate_pirads"],
        sampling_rates=rates,
        est_pos=est_pos,
        cal_pos=cal_pos,
        test_pos=test_pos,
        est_y=est_y,
        cal_y=cal_y,
        test_y=test_y,
        summary=summary,
        alphas=alphas,
        violin_alphas=violin_alphas,
        num_lambdas=num_lambdas,
        delta=float(summary.get("delta", 0.15)),
        utility=str(summary.get("ltt_conditional_utility", "balanced_accuracy_stopped")),
        control_idxs=[0],
        alpha_policy=alpha_policy,
        alpha_per_label_ref=None,
    )


def sweep_cifar(
    summary_path: Path,
    score_cache: Path,
    alphas: Sequence[float],
    violin_alphas: Sequence[float],
    num_lambdas: int,
) -> AlphaSweepResult:
    from eval_compare_cifar10 import select_lambda_methods as select_cifar
    from eval_fst_cifar10 import (
        apply_stopping_rule,
        build_max_score_baseline,
        per_lambda_calibration_arrays,
    )

    summary = json.loads(summary_path.read_text())
    est = torch.load(score_cache / "est.pt", map_location="cpu", weights_only=False)
    cal = torch.load(score_cache / "calib.pt", map_location="cpu", weights_only=False)
    test = torch.load(score_cache / "test.pt", map_location="cpu", weights_only=False)
    rates = [float(r) for r in est["sampling_rates"]]
    est_probs = np.asarray(est["probs"], dtype=float)
    cal_probs = np.asarray(cal["probs"], dtype=float)
    test_probs = np.asarray(test["probs"], dtype=float)
    cal_labels = np.asarray(cal["labels"], dtype=int)

    stopping_rule = str(summary.get("stopping_rule", "max_score"))
    set_size_max = int(summary.get("set_size_max", 1))
    loss_kind = str(summary.get("loss_kind", "argmax_miscoverage"))
    delta_est = float(summary.get("delta_est", 0.05))
    delta = float(summary.get("delta", 0.1))
    utility = str(summary.get("ltt_conditional_utility", "stopped_count"))

    baseline_t = build_max_score_baseline(est_probs, delta_est)
    lambdas = np.linspace(0.0, 1.0, int(num_lambdas))
    losses_mat, stopped_mat, stop_idx_mat = per_lambda_calibration_arrays(
        cal_probs,
        cal_labels,
        lambdas,
        stopping_rule,
        baseline_t,
        set_size_max,
        loss_kind=loss_kind,
    )

    lambda_hats: List[Optional[float]] = []
    stop_fracs: List[List[float]] = []
    mean_rs: List[float] = []
    finals: List[float] = []
    stop_rates_by_alpha: Dict[str, List[float]] = {}
    stopped_mask_by_alpha: Dict[str, List[bool]] = {}
    violin_set = list(violin_alphas)

    for alpha in alphas:
        sel = select_cifar(
            lambdas,
            losses_mat,
            stopped_mat,
            stop_idx_mat,
            float(alpha),
            delta,
            float(summary.get("abstain_penalty", 1.0)),
            str(summary.get("ltt_abstain_utility", "stopped_count")),
            utility,
            None,
            None,
        )
        lam = sel["ltt_conditional"]["lambda_hat"]
        reason = sel["ltt_conditional"]["meta"].get("reason")
        print(
            f"  [cifar10] α={float(alpha):g} → λ̂="
            f"{'None' if lam is None else f'{float(lam):.4f}'} ({reason})",
            flush=True,
        )
        lambda_hats.append(None if lam is None else float(lam))
        if lam is None:
            stop_fracs.append([0.0] * len(rates))
            mean_rs.append(float(rates[-1]))
            finals.append(0.0)
            continue
        thresholds_t = np.clip(float(lam) * baseline_t, 0.0, 1.0)
        stop_idx, stopped = apply_stopping_rule(
            test_probs, stopping_rule, thresholds_t, set_size_max
        )
        frac, mean_r, final = _stop_curve_from_indices(stop_idx, stopped, rates)
        stop_fracs.append(frac)
        mean_rs.append(mean_r)
        finals.append(final)
        if any(abs(float(alpha) - a) < 1e-12 for a in violin_set):
            key = f"{float(alpha):g}"
            rarr = np.asarray(rates, dtype=float)
            stop_rates_by_alpha[key] = rarr[stop_idx].astype(float).tolist()
            stopped_mask_by_alpha[key] = stopped.astype(bool).tolist()

    return AlphaSweepResult(
        dataset="cifar10",
        display_name=DATASET_DISPLAY["cifar10"],
        sampling_rates=rates,
        alphas=[float(a) for a in alphas],
        delta=delta,
        n_test=int(test_probs.shape[0]),
        lambda_hat=lambda_hats,
        stop_frac=stop_fracs,
        mean_r_stop=mean_rs,
        stop_frac_final=finals,
        stop_rates_by_alpha=stop_rates_by_alpha,
        stopped_mask_by_alpha=stopped_mask_by_alpha,
        notes={"stopping_rule": stopping_rule, "num_lambdas": int(num_lambdas)},
    )


def result_from_dict(d: Dict[str, Any]) -> AlphaSweepResult:
    return AlphaSweepResult(
        dataset=str(d["dataset"]),
        display_name=str(d.get("display_name", d["dataset"])),
        sampling_rates=[float(x) for x in d["sampling_rates"]],
        alphas=[float(x) for x in d["alphas"]],
        delta=float(d["delta"]),
        n_test=int(d["n_test"]),
        lambda_hat=[None if x is None else float(x) for x in d["lambda_hat"]],
        stop_frac=[[float(v) for v in row] for row in d["stop_frac"]],
        mean_r_stop=[float(x) for x in d["mean_r_stop"]],
        stop_frac_final=[float(x) for x in d["stop_frac_final"]],
        stop_rates_by_alpha={
            str(k): [float(v) for v in vals]
            for k, vals in (d.get("stop_rates_by_alpha") or {}).items()
        },
        stopped_mask_by_alpha={
            str(k): [bool(v) for v in vals]
            for k, vals in (d.get("stopped_mask_by_alpha") or {}).items()
        },
        notes=dict(d.get("notes") or {}),
    )


def _alpha_colormap(alphas: Sequence[float]):
    """Sequential print-friendly α colors (deep teal → warm coral)."""
    n = max(len(alphas), 1)
    stops = np.array(
        [
            [0.12, 0.27, 0.42],  # deep slate-blue
            [0.18, 0.45, 0.55],  # teal
            [0.25, 0.58, 0.52],  # sea green
            [0.55, 0.68, 0.35],  # olive
            [0.85, 0.55, 0.28],  # amber
            [0.78, 0.30, 0.32],  # coral
        ],
        dtype=float,
    )
    if n == 1:
        return [tuple(stops[2])]
    xs = np.linspace(0.0, 1.0, len(stops))
    ts = np.linspace(0.0, 1.0, n)
    out = []
    for t in ts:
        # Piecewise linear interpolate in RGB.
        i = int(np.searchsorted(xs, t, side="right") - 1)
        i = max(0, min(i, len(stops) - 2))
        u = 0.0 if xs[i + 1] == xs[i] else (t - xs[i]) / (xs[i + 1] - xs[i])
        rgb = (1.0 - u) * stops[i] + u * stops[i + 1]
        out.append(tuple(rgb.tolist()))
    return out


def _style_ax(ax, *, grid: bool = True, grid_y: bool = True) -> None:
    style_compare_ax(ax, grid=grid, grid_y=grid_y)


def _panel_label(ax, letter: str) -> None:
    """Kept for legacy split-figure callers; no-op (panel letters removed)."""
    return None


def _active_alpha_mask(res: AlphaSweepResult) -> List[bool]:
    return [lam is not None for lam in res.lambda_hat]


def plot_publication_composite(
    results: Sequence[AlphaSweepResult],
    out_dir: Path,
    *,
    hide_empty_safe: bool = True,
    include_expected_rate: bool = True,
) -> Path:
    """
    Publication MRI α-sweep panel (no violins, no panel letters).

    Default 2×N:
      top:    stop fraction vs sampling rate (one panel per dataset)
      bottom: expected acquisition E[r_stop] vs α (one panel per dataset)

    With ``include_expected_rate=False``, only the stop-fraction row is drawn
    (saved as ``alpha_stopping_mri_panel_top.png``).

    Dataset names are axes titles; shared α legend sits above them.
    """
    if not results:
        raise ValueError("No results to plot")

    n = len(results)
    ncols = max(n, 1)
    if include_expected_rate:
        # Extra figure height + top margin so the shared legend sits clearly above titles.
        fig = plt.figure(figsize=(6.6 * max(ncols, 2), 8.6), dpi=COMPARE_PUBLICATION_DPI)
        gs = fig.add_gridspec(
            2,
            ncols,
            height_ratios=[1.05, 1.0],
            hspace=0.32,
            wspace=0.36,
        )
        top_adjust = 0.84
        out_name = "alpha_stopping_mri_panel.png"
    else:
        fig = plt.figure(figsize=(6.6 * max(ncols, 2), 4.6), dpi=COMPARE_PUBLICATION_DPI)
        gs = fig.add_gridspec(1, ncols, wspace=0.36)
        top_adjust = 0.78
        out_name = "alpha_stopping_mri_panel_top.png"
    fig.patch.set_facecolor("white")

    legend_res = max(results, key=lambda r: sum(_active_alpha_mask(r)))
    colors = _alpha_colormap(legend_res.alphas)
    active_leg = _active_alpha_mask(legend_res) if hide_empty_safe else [True] * len(legend_res.alphas)

    # Top plot row: stop-fraction panels (dataset names as axes titles)
    axes_stop = []
    for i in range(ncols):
        ax = fig.add_subplot(gs[0, i])
        axes_stop.append(ax)
        if i >= n:
            ax.axis("off")
            continue
        res = results[i]
        mask = _active_alpha_mask(res) if hide_empty_safe else [True] * len(res.alphas)
        for alpha, frac, color, keep in zip(res.alphas, res.stop_frac, colors, mask):
            if not keep:
                continue
            ax.plot(
                res.sampling_rates,
                frac,
                color=color,
                marker="o",
                markersize=6.0,
                markeredgecolor="white",
                markeredgewidth=0.85,
                linewidth=2.35,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
                label=rf"$\alpha={alpha:g}$",
            )
        ax.set_title(
            res.display_name,
            fontsize=16,
            fontweight="semibold",
            color="#1F2933",
            pad=12,
        )
        ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
        if i == 0:
            ax.set_ylabel("Fraction stopped")
        ax.set_ylim(0.0, 1.02)
        xmin, xmax = min(res.sampling_rates), max(res.sampling_rates)
        pad = 0.05 * (xmax - xmin + 1e-9)
        ax.set_xlim(xmin - pad, xmax + pad)
        _style_ax(ax)

    axes_exp = []
    if include_expected_rate:
        # Bottom row: E[r_stop] vs α — per-dataset y-scale so shallow slopes remain readable
        for i in range(ncols):
            ax = fig.add_subplot(gs[1, i])
            axes_exp.append(ax)
            if i >= n:
                ax.axis("off")
                continue
            res = results[i]
            mask = _active_alpha_mask(res) if hide_empty_safe else [True] * len(res.alphas)
            xs = [a for a, keep in zip(res.alphas, mask) if keep]
            ys = [y for y, keep in zip(res.mean_r_stop, mask) if keep]
            color = DATASET_COLORS.get(res.dataset, ALPHA_PALETTE[i % len(ALPHA_PALETTE)])
            if xs:
                ax.fill_between(xs, ys, color=color, alpha=0.12, linewidth=0, zorder=1)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    marker="o",
                    markersize=8.5,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    linewidth=2.6,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=3,
                )
                y_lo = min(ys)
                y_hi = max(ys)
                ypad = max(0.02, 0.18 * (y_hi - y_lo + 1e-9))
                ax.set_ylim(max(0.0, y_lo - ypad), y_hi + ypad)
            ax.set_xlabel(r"Control level $\alpha$")
            if i == 0:
                ax.set_ylabel(r"Expected acquisition $\mathbb{E}[r_{\mathrm{stop}}]$")
            _style_ax(ax)

    handles, labels = [], []
    for alpha, color, keep in zip(legend_res.alphas, colors, active_leg):
        if not keep:
            continue
        (h,) = axes_stop[0].plot(
            [], [], color=color, marker="o", markersize=6.0,
            markeredgecolor="white", markeredgewidth=0.85, linewidth=2.35,
        )
        handles.append(h)
        labels.append(rf"$\alpha={alpha:g}$")
    n_leg = max(1, len(handles))
    # Room above the axes for a compact single-row legend (not colliding with titles).
    fig.subplots_adjust(left=0.08, right=0.98, top=top_adjust, bottom=0.14 if not include_expected_rate else 0.07)
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=n_leg,  # one row — avoids a tall legend box colliding with titles
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#E2E6EA",
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.58,
        handlelength=1.55,
        columnspacing=1.15,
        handletextpad=0.45,
        borderpad=0.45,
    )
    leg.get_frame().set_linewidth(0.8)
    for ax in list(axes_stop) + list(axes_exp):
        if not ax.has_data():
            continue
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, length=3.2)
        for lab in ax.get_yticklabels():
            lab.set_visible(True)
            lab.set_color("#4A5560")
    out = out_dir / out_name
    # Keep reserved top whitespace (tight bbox collapses the legend–title gap).
    save_compare_figure(fig, out, dpi=COMPARE_PUBLICATION_DPI, bbox_inches=None, pad_inches=0.15)
    return out



def plot_stop_frac_small_multiples(
    results: Sequence[AlphaSweepResult],
    out_dir: Path,
    *,
    hide_empty_safe: bool = True,
) -> Path:
    n = len(results)
    fig, axes = plt.subplots(
        1, max(n, 1), figsize=(6.8 * max(n, 1), 5.0), dpi=COMPARE_PUBLICATION_DPI, sharey=True
    )
    axes_list = np.atleast_1d(axes).ravel().tolist()
    for i, (ax, res) in enumerate(zip(axes_list, results)):
        colors = _alpha_colormap(res.alphas)
        mask = _active_alpha_mask(res) if hide_empty_safe else [True] * len(res.alphas)
        for alpha, frac, color, keep in zip(res.alphas, res.stop_frac, colors, mask):
            if not keep:
                continue
            ax.plot(
                res.sampling_rates,
                frac,
                color=color,
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linewidth=2.2,
                label=rf"$\alpha={alpha:g}$",
            )
        ax.set_title(res.display_name)
        ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
        if i == 0:
            ax.set_ylabel("Fraction stopped")
        ax.set_ylim(0.0, 1.02)
        _style_ax(ax)
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.995),
        ncol=min(6, max(1, len(handles))), fontsize=COMPARE_LEGEND_FONTSIZE * 0.75, frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    out = out_dir / "alpha_stop_fraction_vs_rate.png"
    save_compare_figure(fig, out, dpi=COMPARE_PUBLICATION_DPI)
    return out


def plot_mean_r_vs_alpha(
    results: Sequence[AlphaSweepResult],
    out_dir: Path,
    *,
    hide_empty_safe: bool = True,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), dpi=COMPARE_PUBLICATION_DPI)
    markers = ["o", "s", "^", "D", "v"]
    for i, res in enumerate(results):
        mask = _active_alpha_mask(res) if hide_empty_safe else [True] * len(res.alphas)
        xs = [a for a, keep in zip(res.alphas, mask) if keep]
        ys = [y for y, keep in zip(res.mean_r_stop, mask) if keep]
        if not xs:
            continue
        ax.plot(
            xs, ys,
            color=DATASET_COLORS.get(res.dataset, ALPHA_PALETTE[i]),
            marker=markers[i % len(markers)], markersize=8.0,
            markeredgecolor="white", markeredgewidth=0.7, linewidth=2.4,
            label=res.display_name,
        )
    ax.set_xlabel(r"Control level $\alpha$")
    ax.set_ylabel(r"Expected acquisition $\mathbb{E}[r_{\mathrm{stop}}]$")
    ax.legend(fontsize=COMPARE_LEGEND_FONTSIZE * 0.8, frameon=False)
    _style_ax(ax)
    fig.tight_layout()
    out = out_dir / "alpha_expected_rate.png"
    save_compare_figure(fig, out, dpi=COMPARE_PUBLICATION_DPI)
    return out


def plot_stop_time_violins(
    results: Sequence[AlphaSweepResult],
    out_dir: Path,
    *,
    hide_empty_safe: bool = True,
) -> Path:
    usable = [r for r in results if r.stop_rates_by_alpha]
    if not usable:
        raise RuntimeError("No stop-time distributions available for violin plot")
    n = len(usable)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.8), dpi=COMPARE_PUBLICATION_DPI)
    axes_list = np.atleast_1d(axes).ravel().tolist()
    for i, (ax, res) in enumerate(zip(axes_list, usable)):
        keys = sorted(res.stop_rates_by_alpha.keys(), key=lambda s: float(s))
        if hide_empty_safe:
            keys = [
                k for k in keys
                if any(abs(float(k) - a) < 1e-12 and lam is not None for a, lam in zip(res.alphas, res.lambda_hat))
            ]
        data = [np.asarray(res.stop_rates_by_alpha[k], dtype=float) for k in keys]
        if not data:
            ax.set_visible(False)
            continue
        color = DATASET_COLORS.get(res.dataset, "#4C72B0")
        parts = ax.violinplot(data, positions=np.arange(1, len(keys) + 1), showmeans=True, showextrema=False, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("0.25")
            body.set_alpha(0.55)
            body.set_linewidth(0.8)
        if "cmeans" in parts:
            parts["cmeans"].set_color(color)
            parts["cmeans"].set_linewidth(1.6)
        ax.set_xticks(np.arange(1, len(keys) + 1))
        ax.set_xticklabels([rf"$\alpha={k}$" for k in keys])
        ax.set_title(res.display_name)
        ax.set_ylabel("Stop sampling rate")
        ax.set_ylim(0.0, max(1.05 * max(float(np.max(d)) for d in data), 0.05))
        _style_ax(ax)
    fig.tight_layout()
    out = out_dir / "alpha_stop_time_violins.png"
    save_compare_figure(fig, out, dpi=COMPARE_PUBLICATION_DPI)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=str, default="knee_slice,prostate_pirads")
    p.add_argument("--alpha_grid", type=str, default="0.05,0.075,0.1,0.125,0.15,0.2")
    p.add_argument("--violin_alphas", type=str, default="0.1,0.15,0.2")
    p.add_argument("--num_lambdas", type=int, default=51)
    p.add_argument("--alpha_policy", type=str, default="scale", choices=["broadcast", "scale"])
    p.add_argument("--out_dir", type=str, default=str(WORKSPACE / "plots/alpha_stopping_mri"))
    p.add_argument("--from_json", type=str, default=None)
    p.add_argument("--also_split_figures", action="store_true")
    p.add_argument("--show_empty_safe", action="store_true")
    p.add_argument("--knee_summary", type=str, default=str(DEFAULT_PATHS["knee_slice"]["summary"]))
    p.add_argument("--knee_score_cache", type=str, default=str(DEFAULT_PATHS["knee_slice"]["score_cache"]))
    p.add_argument("--cifar_summary", type=str, default=str(DEFAULT_PATHS["cifar10"]["summary"]))
    p.add_argument("--cifar_score_cache", type=str, default=str(DEFAULT_PATHS["cifar10"]["score_cache"]))
    p.add_argument("--imagenet_summary", type=str, default=str(DEFAULT_PATHS["imagenet_hf"]["summary"]))
    p.add_argument("--imagenet_score_cache", type=str, default=str(DEFAULT_PATHS["imagenet_hf"]["score_cache"]))
    p.add_argument("--prostate_summary", type=str, default=str(DEFAULT_PATHS["prostate_pirads"]["summary"]))
    p.add_argument("--prostate_analysis_dir", type=str, default=None)
    p.add_argument("--prostate_val_csv", type=str, default=None)
    p.add_argument("--prostate_test_csv", type=str, default=None)
    p.add_argument(
        "--prostate_score_cache",
        type=str,
        default=str(REPO / "data_cache/prostate_path15_scores.npz"),
    )
    p.add_argument("--prostate_seed", type=int, default=13)
    p.add_argument("--prostate_est_size", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_compare_plot_style(
        **{
            "font.size": 17.0,
            "axes.labelsize": 17.0,
            "axes.titlesize": 17.0,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 15.0,
        }
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hide_empty = not bool(args.show_empty_safe)

    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text())
        want = {d.strip() for d in args.datasets.split(",") if d.strip()}
        results = [
            result_from_dict(r)
            for r in payload["results"]
            if (not want) or r["dataset"] in want
        ]
        if not results:
            raise RuntimeError(f"No matching datasets in {args.from_json} for {want}")
        print(f"Replotting {len(results)} dataset(s) from {args.from_json}", flush=True)
    else:
        alphas = _parse_float_list(args.alpha_grid)
        violin_alphas = _parse_float_list(args.violin_alphas)
        for a in violin_alphas:
            if not any(abs(a - b) < 1e-12 for b in alphas):
                raise ValueError(f"--violin_alphas value {a} is not in --alpha_grid")
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        results = []
        for name in datasets:
            print(f"\n=== Sweeping {name} ===", flush=True)
            if name == "knee_slice":
                results.append(
                    sweep_knee_slice(
                        Path(args.knee_summary), Path(args.knee_score_cache),
                        alphas, violin_alphas, args.num_lambdas, args.alpha_policy,
                    )
                )
            elif name == "cifar10":
                results.append(
                    sweep_cifar(
                        Path(args.cifar_summary), Path(args.cifar_score_cache),
                        alphas, violin_alphas, args.num_lambdas,
                    )
                )
            elif name == "imagenet_hf":
                results.append(
                    sweep_imagenet(
                        Path(args.imagenet_summary), Path(args.imagenet_score_cache),
                        alphas, violin_alphas, args.num_lambdas, args.alpha_policy,
                    )
                )
            elif name == "prostate_pirads":
                results.append(
                    sweep_prostate(
                        Path(args.prostate_summary),
                        Path(args.prostate_analysis_dir) if args.prostate_analysis_dir else None,
                        alphas, violin_alphas, args.num_lambdas, args.alpha_policy,
                        args.prostate_seed, args.prostate_est_size,
                        score_cache=Path(args.prostate_score_cache) if args.prostate_score_cache else None,
                        val_csv=Path(args.prostate_val_csv) if args.prostate_val_csv else None,
                        test_csv=Path(args.prostate_test_csv) if args.prostate_test_csv else None,
                    )
                )
            else:
                raise ValueError(f"Unknown dataset {name!r}")

        payload = {
            "alpha_grid": alphas,
            "violin_alphas": violin_alphas,
            "alpha_policy": args.alpha_policy,
            "num_lambdas": args.num_lambdas,
            "results": [asdict(r) for r in results],
        }
        json_path = out_dir / "alpha_stopping_results.json"
        with open(json_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {json_path}", flush=True)

    panel = plot_publication_composite(results, out_dir, hide_empty_safe=hide_empty)
    print(f"Publication panel: {panel}", flush=True)
    panel_top = plot_publication_composite(
        results, out_dir, hide_empty_safe=hide_empty, include_expected_rate=False
    )
    print(f"Publication panel (stop-frac only): {panel_top}", flush=True)
    if args.also_split_figures:
        p1 = plot_stop_frac_small_multiples(results, out_dir, hide_empty_safe=hide_empty)
        p2 = plot_mean_r_vs_alpha(results, out_dir, hide_empty_safe=hide_empty)
        p3 = plot_stop_time_violins(results, out_dir, hide_empty_safe=hide_empty)
        print(f"Split figures:\n  {p1}\n  {p2}\n  {p3}", flush=True)


if __name__ == "__main__":
    main()
