#!/usr/bin/env python3
"""
PI-RADS prostate ensemble comparison (**asymmetric** τ-dilation variant).

Duplicate of ``eval_compare_prostate_pirads.py`` except effective ambiguity endpoints use
``.compute_bands_asym``: one scalar τ∈[0,1] is scanned on a grid with **distinct** powers for the
lower vs upper dilation of the nominal overlap bands (still **one** trajectory → **O(M)** hypotheses
for Holm, same as symmetric λ).

Reads val/test CSVs from a **hardcoded** ``ANALYSIS_DIR`` under ``summary.json``.
``gamma_lower = gamma_upper = 1`` recovers symmetric ``compute_bounds(τ,·,·)`` from ``eval_fst_knee``.

Raw ``b_pos``, ``b_neg`` are still built by ``compute_bands`` on the estimation set-only.

Diagnostics and JSON fields still use keys ``lambda_hat`` / ``lambda_candidates`` **as τ** for
compat with downstream helpers (values are calibration grid indices along the asymmetric curve).

**Risk controlled on calibration:** FNR only, same wrappers as knee.

Optional ``--print_calib_stop_fnr`` stratification and ``--print_controlled_lambda_bands`` reflect
``.compute_bands_asym``. Plots/json use a stable default ``compare_*_<prostate_pirads_fnr_asym>`` stem so
runs overwrite; pass ``--plot-file-tag-append-gammas`` to encode ``γ_lo/γ_hi`` in filenames again when
archiving grids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from compare_figure_export import configure_compare_plot_style

from eval_compare_knee import (
    LAMBDA_METHODS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    cal_tuned_forced_curves,
    cal_tuned_forced_legend_label,
    casc_curves,
    confidence_baseline_legend_label,
    entropy_baseline_legend_label,
    fixed_threshold_schedule_curves,
    fnr_cap_constrained_forced_cal_curves,
    forced_curves,
    forced_fnr_cal_legend_label,
    fpr_losses_with_abstain_zero,
    parse_alpha_per_label,
    plot_compare_fastmri_panel,
    resolve_alpha_per_label,
    select_lambda_methods,
)
from eval_compare_knee_asym import lambda_method_curves as lambda_method_curves_asym
from eval_fst_knee import (
    add_ambiguity_band_args,
    build_uniform_alpha_schedule,
    compute_bands_from_argparse,
    compute_preds_at_stop,
    compute_sample_fnr_loss,
    compute_sample_fpr_loss,
    compute_stop_indices,
    fnr_losses_with_abstain_zero,
    print_ambiguity_band_diagnostics,
    triage_band_config_dict,
)


# --- Default prostate analysis layout (override with --analysis_dir / --repo_root).
REPO_ROOT = Path("/gpfs/data/chopralab/td2105/prostate_kspace/prostate-kspace-classifier")
DEFAULT_ANALYSIS_DIR = (
    REPO_ROOT
    / "outputs"
    / "analysis"
    / "image_t2_recon_dwi_universal_transfer_espirit_path15_bestcfg_seed3"
)
# Mutable module-level paths set in main() from CLI (kept for helpers that close over them).
ANALYSIS_DIR = DEFAULT_ANALYSIS_DIR
SUMMARY_PATH = ANALYSIS_DIR / "summary.json"

# Columns needed from val/test ensemble CSVs (match pirads_score_matrices / summary exports).
PREDICTION_COLUMNS = ("accession_id", "averaging_tag", "pirads_prob", "pirads_label")


def compute_bands_asym(
    tau: float,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    gamma_lower: float = 1.0,
    gamma_upper: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map frozen overlap bands ``(b_neg, b_pos)`` to effective ``[lower, upper]`` for scalar ``τ``.

    **Not** a second band estimator: underlying ``b_pos``, ``b_neg`` still come from
    ``eval_fst_knee.compute_bands`` on scores. This only defines how ``τ`` traverses ambiguity:

    ``f_lo = τ^{γ_lo}``, ``g_hi = τ^{γ_hi}`` (numpy ``pow``, ``τ`` clipped to ``[0,1]``)::

        lower = f_lo · b_neg,

        upper = g_hi · b_pos + (1 − g_hi) · 1.

    Endpoint behavior::

        • ``τ = 1`` ⇒ ``lower = b_neg``, ``upper = b_pos`` (matches tight overlap band),

        • ``τ = 0`` ⇒ ``lower = 0``, ``upper = 1`` ("full slab" ``[0,1]``, same as symmetric case).

    **Recovery:** ``γ_lo = γ_hi = 1`` is identical to ``eval_fst_knee.compute_bounds(tau, b_pos, b_neg)``.

    **Interpretation:** ``γ_lo`` governs how fast the **negative-side** cutoff moves from 0 down
    toward ``b^-`` when ``τ`` increases; ``γ_hi`` governs how fast the **positive-side** cutoff
    moves from 1 down toward ``b^+``. If ``γ_hi > γ_lo``, the upper face stays nearer 1 longer (harder
    to declare "above band" vs below) relative to how fast the lower face leaves 0—a way to emphasize
    not missing high-confidence negatives vs positives (or vice versa) without an ``M²`` Cartesian
    search that would break FWER bookkeeping.
    """
    tau_cl = float(np.clip(float(tau), 0.0, 1.0))
    gamma_lo = float(gamma_lower)
    gamma_hi = float(gamma_upper)
    if gamma_lo <= 0.0 or gamma_hi <= 0.0:
        raise ValueError(f"gamma_lower and gamma_upper must be positive; got {gamma_lo}, {gamma_hi}")
    f_lo = float(np.power(tau_cl, gamma_lo))
    g_hi = float(np.power(tau_cl, gamma_hi))
    lower = f_lo * b_neg
    upper = g_hi * b_pos + (1.0 - g_hi)
    return lower, upper


def per_lambda_calibration_arrays_prostate_asym(
    pos_scores: np.ndarray,
    labels: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    tau_candidates: np.ndarray,
    gamma_lower: float,
    gamma_upper: float,
    risk_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same contract as knee ``per_lambda_calibration_arrays_knee_metric``, using ``compute_bands_asym``."""
    n_cal = labels.shape[0]
    m = int(tau_candidates.shape[0])
    n_labels = int(labels.shape[1])
    losses_mat = np.zeros((n_cal, m), dtype=float)
    stopped_mat = np.zeros((n_cal, m), dtype=bool)
    stop_idx_mat = np.zeros((n_cal, m), dtype=int)
    preds_cube = np.zeros((n_cal, m, n_labels), dtype=int)

    for j in range(m):
        lower, upper = compute_bands_asym(
            float(tau_candidates[j]),
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


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_prediction_columns(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=PREDICTION_COLUMNS)


def build_pirads_score_matrix(
    predictions: pd.DataFrame,
    averaging_tags: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_tags = [str(tag) for tag in averaging_tags.tolist()]
    frame = predictions.copy()
    frame["averaging_tag"] = frame["averaging_tag"].astype(str)

    duplicates = frame.duplicated(["accession_id", "averaging_tag"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["accession_id", "averaging_tag"]].head()
        raise ValueError(f"Duplicate accession/tag rows found:\n{examples}")

    observed_tags = set(frame["averaging_tag"].unique())
    expected_tag_set = set(expected_tags)
    if observed_tags != expected_tag_set:
        missing = sorted(expected_tag_set - observed_tags)
        extra = sorted(observed_tags - expected_tag_set)
        raise ValueError(f"Tag mismatch. missing={missing} extra={extra}")

    rows_per_accession = frame.groupby("accession_id", sort=False).size()
    if not (rows_per_accession == len(expected_tags)).all():
        bad = rows_per_accession[rows_per_accession != len(expected_tags)].head()
        raise ValueError(f"Expected {len(expected_tags)} rows per accession. Bad rows:\n{bad}")

    label_counts = frame.groupby("accession_id", sort=False)["pirads_label"].nunique(dropna=False)
    if not (label_counts == 1).all():
        bad = label_counts[label_counts != 1].head()
        raise ValueError(f"PI-RADS labels inconsistent across tags:\n{bad}")

    frame["pirads_prob"] = pd.to_numeric(frame["pirads_prob"], errors="coerce")
    probs = frame["pirads_prob"].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError("pirads_prob contains NaN or infinite values.")
    if ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError("pirads_prob contains values outside [0, 1].")

    labels_by_accession = frame.groupby("accession_id", sort=False)["pirads_label"].first()
    if labels_by_accession.isna().any():
        raise ValueError("pirads_label contains missing values.")

    accession_ids = labels_by_accession.index.to_numpy()
    labels = labels_by_accession.astype(np.int64).to_numpy()

    score_table = frame.pivot(
        index="accession_id",
        columns="averaging_tag",
        values="pirads_prob",
    ).reindex(index=accession_ids, columns=expected_tags)
    if score_table.isna().any().any():
        raise ValueError("Score matrix has missing values after pivot/reindex.")

    scores = score_table.to_numpy(dtype=np.float64)
    return scores, labels, accession_ids


def split_val_into_est_calib(
    n_val: int,
    est_size: int,
    calib_fraction: float,
    seed: int,
    use_full_remaining_as_calib: bool,
) -> Tuple[np.ndarray, np.ndarray]:
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
        return est_idx.astype(np.int64), rem.astype(np.int64)

    rng2 = np.random.default_rng(int(seed))
    order = rng2.permutation(rem.size)
    rem_ord = rem[order]
    n_calib = int(np.floor(rem_ord.size * float(calib_fraction)))
    if n_calib <= 0:
        raise ValueError(
            f"calibration split empty (n_rem={rem_ord.size}, calib_fraction={calib_fraction})"
        )
    return est_idx.astype(np.int64), rem_ord[:n_calib].astype(np.int64)


def pirads_sampling_rates(tags: np.ndarray) -> Tuple[List[float], List[str]]:
    """Synthetic x-axis: fraction along the fixed tag order (aligned with ambiguity band dimension T)."""
    t = len(tags)
    rates = [(k + 1) / float(t) for k in range(t)]
    xtick_labels = [str(tags[k]) for k in range(t)]
    return rates, xtick_labels


def resolve_calibration_diagnostic_lambda(
    override: Optional[float],
    method_selections: Dict[str, Dict],
) -> Tuple[Optional[float], str]:
    """Pick λ for stratified diagnostics: CLI override > LTT-cond."""
    if override is not None:
        return float(override), "--calib_stop_fnr_lambda"
    order = ["ltt_conditional"]
    for key in order:
        lam = method_selections[key]["lambda_hat"]
        if lam is not None:
            return float(lam), key
    return None, ""


def collect_and_print_effective_lambda_bands(
    method_selections: Dict[str, Dict],
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    tag_labels_for_plot: List[str],
    sampling_rates: List[float],
    gamma_lower: float,
    gamma_upper: float,
) -> Dict[str, object]:
    """
    Show ``compute_bands_asym(tau_hat,…)`` effective ``[lower,upper]`` per timestep per method.

    ``lambda_hat`` in ``method_selections`` is the calibration gridordinate ``τ``.
    """
    t_steps, n_lab = b_pos.shape
    if n_lab != 1:
        raise ValueError(f"Effective-band printout is K=1-only; got n_labels={n_lab}.")

    print(
        "[effective_bands_asym] Per (t,k): lower = τ^{γ_lo}·b_neg, "
        "upper = τ^{γ_hi}·b_pos + (1−τ^{γ_hi}); "
        "ambiguous at t iff score ∈ [lower,upper]. "
        f"γ_lo={gamma_lower:g}, γ_hi={gamma_upper:g}. "
        "Blocks use each method's selected τ (= stored lambda_hat).\n",
        flush=True,
    )

    report: Dict[str, object] = {}
    for key in LAMBDA_METHODS:
        sel = method_selections[key]
        lam = sel["lambda_hat"]
        reason = str(sel["meta"]["reason"])
        pretty = METHOD_LABELS.get(key, key)
        if lam is None:
            print(
                f"  {pretty}: lambda_hat=None  (selection: {reason})\n",
                flush=True,
            )
            report[key] = {
                "label": pretty,
                "lambda_hat": None,
                "tau_hat_equals_lambda_hat_slot": None,
                "selection_reason": reason,
                "per_timestep": [],
            }
            continue

        tau_hat = float(lam)
        lower, upper = compute_bands_asym(
            tau_hat,
            b_pos,
            b_neg,
            gamma_lower,
            gamma_upper,
        )
        per_t: List[Dict[str, object]] = []
        print(
            f"  {pretty}: tau_hat={tau_hat:.6g} (stored as lambda_hat)  (selection: {reason})",
            flush=True,
        )
        hdr = f"    {'t':>3}  {'sr':>8}  {'lower':>10}  {'upper':>10}  {'width':>10}  tag"
        print(hdr, flush=True)
        print(f"    {'-' * (len(hdr) - 4)}", flush=True)
        for t in range(t_steps):
            lo = float(lower[t, 0])
            hi = float(upper[t, 0])
            tag = tag_labels_for_plot[t] if t < len(tag_labels_for_plot) else str(t)
            sr = (
                float(sampling_rates[t])
                if t < len(sampling_rates)
                else float((t + 1) / max(t_steps, 1))
            )
            w = hi - lo
            inv = "  (INVERTED lo>hi)" if lo > hi else ""
            per_t.append(
                {
                    "t": int(t),
                    "synthetic_rate": sr,
                    "lower": lo,
                    "upper": hi,
                    "width": float(w),
                    "tag": str(tag),
                    "inverted": bool(lo > hi),
                }
            )
            print(
                f"    {t:3d}  {sr:8.5f}  {lo:10.6f}  {hi:10.6f}  {w:10.6f}  {tag}{inv}",
                flush=True,
            )
        print("", flush=True)
        report[key] = {
            "label": pretty,
            "lambda_hat": tau_hat,
            "tau_hat_equals_lambda_hat_slot": tau_hat,
            "gamma_lower": float(gamma_lower),
            "gamma_upper": float(gamma_upper),
            "selection_reason": reason,
            "per_timestep": per_t,
        }

    return report


def run_calib_stop_stratified_fnr_diagnostic(
    lam: Optional[float],
    lambda_source: str,
    alpha: float,
    calib_pos_scores: np.ndarray,
    label_matrix_cal: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    tag_labels_for_plot: List[str],
    sampling_rates: List[float],
    gamma_lower: float,
    gamma_upper: float,
) -> Optional[Dict]:
    """
    Calibration-only stratification using ``compute_bands_asym`` with the supplied ``γ``.
    """
    if lam is None:
        print(
            "[calib_stop_fnr] skipped: no diagnostic lambda "
            "(LTT-cond / LTT-abstain / FST lambda_hat all None).\n",
            flush=True,
        )
        return None

    _, t_steps, n_lab = calib_pos_scores.shape
    if n_lab != 1:
        raise ValueError(f"Diagnostic implemented for K=1 only; got n_labels={n_lab}.")

    lower, upper = compute_bands_asym(
        float(lam),
        b_pos,
        b_neg,
        gamma_lower,
        gamma_upper,
    )
    stop_indices, stopped = compute_stop_indices(calib_pos_scores, lower, upper)
    preds = compute_preds_at_stop(calib_pos_scores, stop_indices, b_pos, b_neg)
    losses_all = compute_sample_fnr_loss(preds, label_matrix_cal)
    y = label_matrix_cal[:, 0].astype(np.int64, copy=False)
    n_cal = int(label_matrix_cal.shape[0])

    pos_stopped = np.logical_and(stopped, y == 1)
    pooled_stopped_pos_fnr = (
        float(np.mean(losses_all[pos_stopped])) if np.any(pos_stopped) else float("nan")
    )

    def _never_cleared_row() -> Dict:
        m = np.logical_and(~stopped, y == 1)
        nn = np.logical_and(~stopped, y == 0)
        fnr_pos = float(np.mean(losses_all[m])) if np.any(m) else float("nan")
        return {
            "bucket": "never_cleared",
            "stop_index": None,
            "tag": None,
            "synthetic_rate": None,
            "n_samples": int((~stopped).sum()),
            "n_positive": int(m.sum()),
            "n_negative": int(nn.sum()),
            "mean_fnr_on_positives": fnr_pos,
            "fnr_gt_alpha_positives": (
                None
                if not np.isfinite(fnr_pos)
                else bool(fnr_pos > float(alpha))
            ),
        }

    rows: List[Dict] = []
    if (~stopped).any():
        rows.append(_never_cleared_row())

    for t in range(t_steps):
        mask = np.logical_and(stopped, stop_indices == t)
        idx_sub = mask
        cnt = int(idx_sub.sum())
        if cnt == 0:
            continue
        y_b = y[idx_sub]
        n_pos = int(np.sum(y_b == 1))
        n_neg = int(np.sum(y_b == 0))
        lbl = tag_labels_for_plot[t] if t < len(tag_labels_for_plot) else str(t)
        sr = sampling_rates[t] if t < len(sampling_rates) else float((t + 1) / max(t_steps, 1))

        lb = losses_all[idx_sub]
        if n_pos > 0:
            pos_m = y_b == 1
            fnr_on_pos = float(np.mean(lb[pos_m]))
        else:
            fnr_on_pos = float("nan")
        rows.append(
            {
                "bucket": "stopped_at_timestep",
                "stop_index": int(t),
                "tag": str(lbl),
                "synthetic_rate": float(sr),
                "n_samples": cnt,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "mean_fnr_on_positives": fnr_on_pos,
                "fnr_gt_alpha_positives": (
                    None
                    if not np.isfinite(fnr_on_pos)
                    else bool(fnr_on_pos > float(alpha))
                ),
            }
        )

    payload: Dict[str, object] = {
        "tau_used_equals_lambda_hat_slot": float(lam),
        "lambda_resolution_source": lambda_source,
        "gamma_lower": float(gamma_lower),
        "gamma_upper": float(gamma_upper),
        "control_alpha": float(alpha),
        "n_calibration": n_cal,
        "n_never_cleared": int((~stopped).sum()),
        "pooled_mean_fnr_among_positive_stopped": pooled_stopped_pos_fnr,
        "strata": rows,
    }

    print("", flush=True)
    print(
        "[calib_stop_fnr] stratification @ "
        f"tau={lam:.6g} (lambda_hat slot, from {lambda_source}); γ_lo={gamma_lower:g}, γ_hi={gamma_upper:g}; "
        f"pooled mean FNR on positives (stopped only): {pooled_stopped_pos_fnr:.6g}; "
        f"alpha={alpha:.6g}; n_cal={n_cal}; never_cleared="
        f"{int((~stopped).sum())}",
        flush=True,
    )
    hdr = (
        f"{'bucket':<24}{'ix':>4}{'n':>6}{'pos':>6}{'neg':>6}"
        f"{'fnr@pos':>10}{'>a?':>5}  tag / note"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in rows:
        bnk = row["bucket"]
        ix = row["stop_index"]
        ix_s = "— " if ix is None else f"{ix:>3}"
        sr = row.get("synthetic_rate")
        tag_note = ""
        if row["tag"] is not None:
            tag_note = str(row["tag"])
            if sr is not None:
                tag_note += f" (sr={sr:.4g})"
        fnr_val = row["mean_fnr_on_positives"]
        fnr_s = (
            "     —   "
            if not np.isfinite(fnr_val)
            else f"{fnr_val:>9.4f}"
        )
        ga = row["fnr_gt_alpha_positives"]
        gt_s = "  — " if ga is None else ("  Y " if ga else "  N ")
        print(
            f"{bnk:<24}{ix_s}"
            f"{row['n_samples']:>6}{row['n_positive']:>6}{row['n_negative']:>6}"
            f"{fnr_s}{gt_s}  {tag_note}",
            flush=True,
        )
    print("", flush=True)

    return payload


def main() -> None:
    global REPO_ROOT, ANALYSIS_DIR, SUMMARY_PATH

    parser = argparse.ArgumentParser(
        description=(
            "PI-RADS ensemble: knee-style ambiguity bands + conformal wrappers (FNR control). "
            "Override paths with --repo_root / --analysis_dir."
        )
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(REPO_ROOT),
        help="Prostate-kspace-classifier repo root (relative CSV paths resolve against this).",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Analysis folder containing summary.json (+ ensemble prediction CSVs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./neurips_plots_final/prostate_pirads_compare_asym",
        help="Default separate from symmetric script to avoid clobbering PNGs / JSON.",
    )
    parser.add_argument(
        "--score_cache_base",
        type=str,
        default="/gpfs/data/chopralab/us453/icml_26/paper_release/neurips_plots/prostate_pirads_compare_asym/testset",
        help=(
            "Unused: this script does not use a score_cache directory. "
            "Accepted for CLI parity with other eval_compare_* scripts."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.125, help="Target α (broadcast unless --alpha_per_label).")
    parser.add_argument(
        "--alpha_per_label",
        type=str,
        default=None,
        help="Comma-separated per-label α_k (length K); overrides broadcasting of --alpha for LTT.",
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--num_deltas", type=int, default=101)
    parser.add_argument("--est_size", type=int, default=200)
    parser.add_argument(
        "--calib_fraction",
        type=float,
        default=1.0,
        help="Fraction of val-after-est for calibration when <1 (rest discarded); 1.0 = full remainder.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_lambdas", type=int, default=51)
    parser.add_argument(
        "--band_gamma_lower",
        type=float,
        default=1.0,
        help="Exponent γ_lo on τ for lower dilation: lower = τ^γ_lo · b_neg (γ=1 ⇒ symmetric tau). Must be >0.",
    )
    parser.add_argument(
        "--band_gamma_upper",
        type=float,
        default=1.0,
        help="Exponent γ_hi on τ for upper dilation: upper = τ^γ_hi · b_pos + (1−τ^γ_hi). Must be >0.",
    )
    parser.add_argument(
        "--positive_class_idx",
        type=int,
        default=1,
        help="Index of positive singleton in casc dual-threshold probs stack [neg, pos].",
    )
    parser.add_argument(
        "--abstain_penalty",
        type=float,
        default=1,
        help="LTT-abstain loss on abstained calibration samples.",
    )
    parser.add_argument(
        "--ltt_abstain_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
    )
    parser.add_argument(
        "--ltt_conditional_utility",
        type=str,
        default="stopped_count",
        choices=["stopped_count", "neg_mean_stop_index", "neg_calib_risk", "balanced_accuracy_stopped"],
    )
    parser.add_argument("--default_lambda_idx_abstain", type=int, default=None)
    parser.add_argument("--default_lambda_idx_conditional", type=int, default=None)
    parser.add_argument(
        "--ltt_conditional_test",
        type=str,
        default="ratio",
        choices=["ratio", "via_W"],
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
        help="Cal-tuned forced: each label threshold maximizes this on calibration at that step (default: accuracy).",
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
        help="Add FN-capped forced baseline tuned on calibration positives.",
    )
    parser.add_argument(
        "--forced_fnr_cal_alpha",
        type=float,
        default=None,
        help="FNR cap for that baseline (defaults to --alpha).",
    )
    parser.add_argument(
        "--forced_fnr_match_casc_schedule",
        action="store_true",
        help="Use α/T per timestep for the FN cap.",
    )
    parser.add_argument(
        "--forced_fnr_cal_grid",
        type=int,
        default=101,
        help="θ grid for FN-capped forced search.",
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
        help="Optional baseline: first rate where all per-label positive scores ≥ this threshold.",
    )
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=0.2,
        help="Optional baseline: first rate where mean label entropy ≤ this value.",
    )
    parser.add_argument(
        "--print_calib_stop_fnr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print calibration stratification: first-clear timestep vs mean FNR on positive "
            "labels (--calib_stop_fnr_lambda overrides λ; else use LTT-cond, then abstain, then FST "
            "lambda_hat)."
        ),
    )
    parser.add_argument(
        "--calib_stop_fnr_lambda",
        type=float,
        default=None,
        help="Optional λ override for --print-calib-stop-fnr stratification.",
    )
    parser.add_argument(
        "--print_ambiguity_bands",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--print_controlled_lambda_bands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After FST/LTT selection, print effective [lower,upper] per timestep for each method's "
            "tau_hat (asymmetric compute_bands_asym; scores inside stay ambiguous)."
        ),
    )
    parser.add_argument(
        "--plot_file_tag",
        type=str,
        default=None,
        help=(
            "Stem suffix for compare_*_<tag>.png and summary JSON "
            '(default "prostate_pirads_<risk>_asym"; override for custom naming).'
        ),
    )
    parser.add_argument(
        "--plot_file_tag_append_gammas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append encoded γ_lo/γ_hi to the file stem (old behavior; disables overwrite across gammas).",
    )
    add_ambiguity_band_args(parser)
    args = parser.parse_args()
    configure_compare_plot_style()

    REPO_ROOT = Path(args.repo_root).expanduser().resolve()
    ANALYSIS_DIR = Path(args.analysis_dir).expanduser().resolve()
    SUMMARY_PATH = ANALYSIS_DIR / "summary.json"

    risk_metric = "fnr"

    print(
        f"eval_compare_prostate_pirads_asym (asymmetric τ-dilation γ_lo={args.band_gamma_lower:g}, "
        f"γ_hi={args.band_gamma_upper:g}; γ_lo=γ_hi=1 ⇒ same as symmetric compute_bounds)",
        flush=True,
    )
    print(f"analysis_dir: {ANALYSIS_DIR}")
    print(f"summary_path: {SUMMARY_PATH}")
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Missing summary.json: {SUMMARY_PATH}")

    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    averaging_tags = np.array(summary["averaging_tags"], dtype=str)

    val_predictions_path = resolve_repo_path(summary["outputs"]["val_ensemble_predictions"])
    test_predictions_path = resolve_repo_path(summary["outputs"]["test_ensemble_predictions"])

    print(f"num_averaging_tags: {len(averaging_tags)}")
    print(f"val_predictions_path: {val_predictions_path}")
    print(f"test_predictions_path: {test_predictions_path}")
    val_predictions = load_prediction_columns(val_predictions_path)
    test_predictions = load_prediction_columns(test_predictions_path)

    val_scores, val_labels_scalar, val_ids = build_pirads_score_matrix(val_predictions, averaging_tags)
    test_scores, test_labels_scalar, test_ids = build_pirads_score_matrix(test_predictions, averaging_tags)

    for name, lbl in ("val", val_labels_scalar), ("test", test_labels_scalar):
        u = np.unique(lbl.astype(np.int64))
        if set(u.tolist()) - {0, 1}:
            raise ValueError(f"{name} pirads_label must be binary {{0,1}}; observed {u}")

    label_matrix_val = val_labels_scalar.astype(np.int64)[:, np.newaxis]
    label_matrix_test = test_labels_scalar.astype(np.int64)[:, np.newaxis]

    n_val = val_scores.shape[0]
    use_full_calib = args.calib_fraction >= 1.0 - 1e-12
    est_idx, calib_idx = split_val_into_est_calib(
        n_val, args.est_size, args.calib_fraction, args.seed, use_full_calib
    )

    est_pos_scores = np.ascontiguousarray(val_scores[est_idx, :, np.newaxis], dtype=np.float64)
    calib_pos_scores = np.ascontiguousarray(val_scores[calib_idx, :, np.newaxis], dtype=np.float64)
    label_matrix_est = label_matrix_val[est_idx]
    label_matrix_cal = label_matrix_val[calib_idx]
    test_pos_scores = np.ascontiguousarray(test_scores[:, :, np.newaxis], dtype=np.float64)

    sampling_rates, tag_labels_for_plot = pirads_sampling_rates(averaging_tags)
    lambda_candidates = np.linspace(0.0, 1.0, args.num_lambdas)

    b_pos, b_neg = compute_bands_from_argparse(est_pos_scores, label_matrix_est, args)
    if args.print_ambiguity_bands:
        print_ambiguity_band_diagnostics(
            sampling_rates,
            b_pos,
            b_neg,
            ["pirads_positive"],
            args.num_deltas,
            band_source=str(args.band_source),
        )

    losses_mat, stopped_mat, stop_idx_mat, preds_cube = per_lambda_calibration_arrays_prostate_asym(
        calib_pos_scores,
        label_matrix_cal,
        b_pos,
        b_neg,
        lambda_candidates,
        args.band_gamma_lower,
        args.band_gamma_upper,
        risk_metric,
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
        risk_metric=str(risk_metric),
        alpha_per_label=resolve_alpha_per_label(
            float(args.alpha),
            parse_alpha_per_label(args.alpha_per_label),
            int(label_matrix_cal.shape[1]),
        ),
        pvalue_method=str(getattr(args, "ltt_pvalue_method", "binomial")),
        control_both=bool(getattr(args, "control_fnr_and_fpr", False)),
    )

    effective_bands_report: Optional[Dict[str, object]] = None
    if args.print_controlled_lambda_bands:
        effective_bands_report = collect_and_print_effective_lambda_bands(
            method_selections,
            b_pos,
            b_neg,
            tag_labels_for_plot,
            sampling_rates,
            args.band_gamma_lower,
            args.band_gamma_upper,
        )

    calib_diag = None
    if args.print_calib_stop_fnr:
        diag_lam, diag_src = resolve_calibration_diagnostic_lambda(
            args.calib_stop_fnr_lambda,
            method_selections,
        )
        calib_diag = run_calib_stop_stratified_fnr_diagnostic(
            diag_lam,
            diag_src,
            args.alpha,
            calib_pos_scores,
            label_matrix_cal,
            b_pos,
            b_neg,
            tag_labels_for_plot,
            sampling_rates,
            args.band_gamma_lower,
            args.band_gamma_upper,
        )

    m_grid = int(lambda_candidates.shape[0])
    print(
        f"prostate_pirads_asym risk_metric={risk_metric} (FNR calibration) "
        f"T_steps={averaging_tags.size} tau_grid_pts={m_grid} "
        f"γ_lo={args.band_gamma_lower:g} γ_hi={args.band_gamma_upper:g}"
    )

    curves_by_method: Dict[str, Dict[str, List[float]]] = {}
    for key in LAMBDA_METHODS:
        curves_by_method[key] = lambda_method_curves_asym(
            method_selections[key]["lambda_hat"],
            test_pos_scores,
            label_matrix_test,
            b_pos,
            b_neg,
            args.band_gamma_lower,
            args.band_gamma_upper,
            label_names=["pirads_positive"],
        )
    # Nominal band baseline omitted from plots / summary for now.
    # curves_by_method["nominal_bands"] = lambda_method_curves_asym(
    #     1.0,
    #     test_pos_scores,
    #     label_matrix_test,
    #     b_pos,
    #     b_neg,
    #     args.band_gamma_lower,
    #     args.band_gamma_upper,
    # )

    casc_curve_data = casc_curves(
        calib_pos_scores,
        label_matrix_cal,
        test_pos_scores,
        label_matrix_test,
        args.alpha,
        args.positive_class_idx,
        label_names=["pirads_positive"],
    )
    forced_curve_data = forced_curves(
        test_pos_scores, label_matrix_test, label_names=["pirads_positive"]
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
            label_names=["pirads_positive"],
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
            label_names=["pirads_positive"],
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
        label_names=["pirads_positive"],
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

    tag_custom = (
        args.plot_file_tag.strip() if args.plot_file_tag and str(args.plot_file_tag).strip() else ""
    )
    file_tag_base = tag_custom if tag_custom else f"prostate_pirads_{risk_metric}_asym"
    if args.plot_file_tag_append_gammas:
        _gl_tag = (
            format(float(args.band_gamma_lower), ".4f").replace(".", "p").replace("-", "m")
        )
        _gu_tag = (
            format(float(args.band_gamma_upper), ".4f").replace(".", "p").replace("-", "m")
        )
        file_tag = f"{file_tag_base}_gl{_gl_tag}_gu{_gu_tag}"
    else:
        file_tag = file_tag_base

    print(f"[plots/output] compare_*_{file_tag}.png & summary JSON in {args.output_dir}/fnr/", flush=True)

    out_dir = Path(args.output_dir) / "fnr"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_test = int(label_matrix_test.shape[0])

    plot_compare_fastmri_panel(
        out_dir,
        sampling_rates,
        curves_by_method,
        casc_curve_data,
        forced_curve_data,
        file_tag,
        n_test,
        risk_metric=risk_metric,
        control_alpha=float(
            np.max(
                resolve_alpha_per_label(
                    float(args.alpha),
                    parse_alpha_per_label(args.alpha_per_label),
                    int(label_matrix_cal.shape[1]),
                )
            )
        ),
        cal_tuned_forced_curve=cal_tuned_curve,
        cal_tuned_forced_legend=cal_tuned_legend,
        forced_fnr_cal_curve=forced_fnr_curve,
        forced_fnr_cal_legend=forced_fnr_legend,
        confidence_threshold_curve=conf_thr_curve,
        confidence_threshold_legend=conf_thr_legend,
        entropy_threshold_curve=ent_thr_curve,
        entropy_threshold_legend=ent_thr_legend,
        plot_conditional_middle=bool(args.plot_conditional),
        control_label_names=["pirads_positive"],
        label_prevalence_rows=[
            (
                "pirads_positive",
                int(np.sum(label_matrix_test[:, 0] == 1)),
                int(label_matrix_test.shape[0]),
            )
        ],
        label_prevalence_split="test",
    )

    out_summary = {
        "variant": "prostate_pirads_asymmetric_tau_band",
        "plot_conditional": bool(args.plot_conditional),
        "plot_file_tag_stem": file_tag,
        "plot_file_tag_append_gammas": bool(args.plot_file_tag_append_gammas),
        "band_gamma_lower": float(args.band_gamma_lower),
        "band_gamma_upper": float(args.band_gamma_upper),
        "tau_grid_candidates": lambda_candidates.tolist(),
        "dataset": "prostate_pirads",
        "repo_root": str(REPO_ROOT.resolve()),
        "analysis_dir": str(ANALYSIS_DIR.resolve()),
        "summary_json": str(SUMMARY_PATH.resolve()),
        "risk_metric": risk_metric,
        "alpha": args.alpha,
        "delta": args.delta,
        "averaging_tags": averaging_tags.tolist(),
        "n_val": int(n_val),
        "n_est": int(est_idx.size),
        "n_cal": int(calib_idx.size),
        "n_test": n_test,
        "abstain_penalty": float(args.abstain_penalty),
        "ltt_abstain_utility": args.ltt_abstain_utility,
        "ltt_conditional_utility": args.ltt_conditional_utility,
        "ltt_conditional_test": args.ltt_conditional_test,
        "confidence_threshold": args.confidence_threshold,
        "entropy_threshold": args.entropy_threshold,
        "ambiguity_band": triage_band_config_dict(args),
        "selections": {
            **{
                key: {
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
                for key, sel in method_selections.items()
            },
            # "nominal_bands": {
            #     "lambda_hat": 1.0,
            #     "reason": "fixed_nominal_overlap_from_estimation",
            # },
        },
        "curves": {
            "sampling_rates": sampling_rates,
            "tag_labels": tag_labels_for_plot,
            **curves_by_method,
            "casc": casc_curve_data,
            "forced": forced_curve_data,
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

    if calib_diag is not None:
        out_summary["calib_stop_fnr_stratification"] = calib_diag
    if effective_bands_report is not None:
        out_summary["effective_lambda_bands"] = effective_bands_report

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj))

    out_path = out_dir / f"compare_prostate_pirads_summary_{file_tag}.json"
    with open(out_path, "w") as handle:
        json.dump(out_summary, handle, indent=2, default=_json_default)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
