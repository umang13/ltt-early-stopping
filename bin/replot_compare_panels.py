#!/usr/bin/env python3
"""
Self-contained replot of class-conditional compare panels with all baselines.

Does not import the heavy classifier / Lightning stack — only numpy, torch (for
score caches), matplotlib, and compare_figure_export.

Knee: summary curves for RCS/Forced/cal-tuned + Confidence/Entropy from score cache.
Prostate: all curves from a summary that already stores Confidence/Entropy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_figure_export import (
    COMPARE_FIGSIZE_2x2,
    COMPARE_FIGSIZE_3COL,
    COMPARE_FIGSIZE_KNEE_1x3,
    COMPARE_LEGEND_FONTSIZE,
    COMPARE_RCS_LEGEND_LABEL,
    COMPARE_XLABEL_SAMPLING,
    configure_compare_plot_style,
    draw_alpha_reference,
    plot_compare_series,
    save_compare_figure,
    style_compare_ax,
)

FORCED_PRED_THRESHOLD = 0.5
FASTMRI_PANEL_LAYOUT_TOP = 0.88  # compact headroom for single-row legend

LABEL_DISPLAY = {
    "acl": "ACL",
    "mtear": "Meniscus",
    "abnormal": "Abnormal",
    "pirads_positive": r"PI-RADS $\geq$ 3",
}

METHOD_COLORS = {
    "ltt_conditional": "#8172B2",
    "forced": "#7F7F7F",
    "cal_tuned_forced": "#B07C9E",
    "confidence_threshold": "#BCBD22",
    "entropy_threshold": "#17BECF",
}
METHOD_LINESTYLES = {
    "ltt_conditional": "--",
    "forced": "-",
    "cal_tuned_forced": "--",
    "confidence_threshold": ":",
    "entropy_threshold": ":",
}
METHOD_MARKERS = {
    "ltt_conditional": "D",
    "forced": "o",
    "cal_tuned_forced": "s",
    "confidence_threshold": "v",
    "entropy_threshold": "^",
}


def _label(name: str) -> str:
    return LABEL_DISPLAY.get(str(name), str(name))


def _forced_label() -> str:
    return r"Forced ($p^+ > 0.5$)"


def _cal_tuned_label() -> str:
    return r"Forced ($p^+ > \theta^*$)"


def _conf_label(thr: float) -> str:
    return f"Confidence ≥ {float(thr):g}"


def _ent_label(thr: float) -> str:
    return f"Entropy ≤ {float(thr):g}"


def _style_ax(ax: Axes) -> None:
    style_compare_ax(ax, grid=True)


def _plot(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    key: str,
    label: str,
    zorder: int = 2,
) -> None:
    plot_compare_series(
        ax,
        xs,
        ys,
        color=METHOD_COLORS[key],
        linestyle=METHOD_LINESTYLES[key],
        marker=METHOD_MARKERS[key],
        label=label,
        zorder=zorder,
    )


def _merge_legend(axes: List[Axes], priority: List[str]):
    by: Dict[str, Any] = {}
    for ax in axes:
        hs, ls = ax.get_legend_handles_labels()
        for h, lab in zip(hs, ls):
            if lab not in by:
                by[lab] = h
    hs, ls = [], []
    for lab in priority:
        if lab in by:
            hs.append(by[lab])
            ls.append(lab)
    for lab, h in by.items():
        if lab not in ls:
            hs.append(h)
            ls.append(lab)
    return hs, ls


def _panel_ylim(
    primary_series: List[Sequence[float]],
    forced_series: Optional[Sequence[float]],
    alpha: float,
) -> Tuple[float, float]:
    """Include Forced fully so it is not clipped at the top of the panel."""
    primary = [float(v) for s in primary_series for v in s if np.isfinite(float(v))]
    forced = (
        [float(v) for v in forced_series if np.isfinite(float(v))]
        if forced_series is not None
        else []
    )
    y_pri = max(primary) if primary else 0.0
    y_f = max(forced) if forced else 0.0
    ca = float(alpha)
    y_top = max(y_pri, y_f, ca, 0.08)
    y_top = max(y_top * 1.08, ca * 1.5 if ca > 0 else y_top)
    return 0.0, min(1.05, y_top)


def _fnr_on_idx(preds: np.ndarray, labels: np.ndarray, idx: np.ndarray, lab: int) -> float:
    if idx.size == 0:
        return 0.0
    y = labels[idx, lab]
    p = preds[idx, lab]
    pos = y == 1
    n = int(pos.sum())
    if n == 0:
        return 0.0
    return float(np.sum((p[pos] == 0)) / n)


def _ba_on_idx(preds: np.ndarray, labels: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    bas = []
    for k in range(labels.shape[1]):
        y = labels[idx, k].astype(int)
        p = preds[idx, k].astype(int)
        tp = int(np.sum((p == 1) & (y == 1)))
        tn = int(np.sum((p == 0) & (y == 0)))
        fp = int(np.sum((p == 1) & (y == 0)))
        fn = int(np.sum((p == 0) & (y == 1)))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bas.append(0.5 * (sens + spec))
    return float(np.mean(bas)) if bas else 0.0


def _heuristic_curves(
    test_pos: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str,
    threshold: float,
    control_idxs: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, Any]:
    n, t_steps, k = test_pos.shape
    first = np.full(n, -1, dtype=int)
    thr = float(threshold)
    for t in range(t_steps):
        if mode == "confidence":
            ok = np.all(test_pos[:, t, :] >= thr, axis=1)
        else:
            p = np.clip(test_pos[:, t, :], 1e-12, 1.0 - 1e-12)
            h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).mean(axis=1)
            ok = h <= thr
        m = (first < 0) & ok
        first[m] = t

    stopped = first >= 0
    preds = np.zeros((n, k), dtype=int)
    if np.any(stopped):
        ii = np.where(stopped)[0]
        ts = first[ii]
        for lab in range(k):
            preds[ii, lab] = (test_pos[ii, ts, lab] >= FORCED_PRED_THRESHOLD).astype(int)

    ctrl_names = [label_names[i] for i in control_idxs]
    out: Dict[str, Any] = {
        "stopped_count": [],
        "fnr_cumulative": [],
        "balanced_accuracy_cumulative": [],
        "fnr_cumulative_per_label": {n: [] for n in ctrl_names},
    }
    for t in range(t_steps):
        idx = np.where(stopped & (first <= t))[0]
        out["stopped_count"].append(int(idx.size))
        # Aggregate max FNR over controlled labels (for 1×3).
        fnrs = [_fnr_on_idx(preds, labels, idx, int(i)) for i in control_idxs]
        out["fnr_cumulative"].append(float(max(fnrs)) if fnrs else 0.0)
        out["balanced_accuracy_cumulative"].append(_ba_on_idx(preds, labels, idx))
        for i, name in zip(control_idxs, ctrl_names):
            out["fnr_cumulative_per_label"][name].append(_fnr_on_idx(preds, labels, idx, int(i)))
    return out


def _torch_load(path: Path):
    """Load score-cache pickles across torch versions (2.6+ defaults weights_only=True)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch (<2.0) has no weights_only kwarg.
        return torch.load(path, map_location="cpu")


def _labels_from_cache(raw, names: Sequence[str], pos_idx: int) -> np.ndarray:
    cols = []
    for n in names:
        arr = np.asarray(raw[n])
        if arr.ndim == 1:
            cols.append((arr == pos_idx).astype(np.int64))
        else:
            cols.append(arr[:, pos_idx].astype(np.int64))
    return np.stack(cols, axis=1)


def _probs_from_cache(raw, names: Sequence[str], pos_idx: int) -> np.ndarray:
    cols = []
    for n in names:
        arr = np.asarray(raw[n], dtype=float)
        if arr.ndim == 1:
            cols.append(arr)
        else:
            cols.append(arr[:, pos_idx])
    return np.stack(cols, axis=1)


def _load_score_cubes(score_cache_dir: Path, sampling_rates: Sequence[float]):
    """Return est/cal/test pos-score cubes and label matrices."""
    est_list, cal_list, tst_list = [], [], []
    y_est = y_cal = y_tst = None
    names = None
    pos_idx = 1
    for sr in sampling_rates:
        path = score_cache_dir / f"sr_{float(sr):.6f}.pt"
        cached = _torch_load(path)
        if names is None:
            names = list(cached["label_names"])
            pos_idx = int(cached.get("positive_class_idx", 1))
            y_est = _labels_from_cache(cached["est_labels"], names, pos_idx)
            y_cal = _labels_from_cache(cached["cal_labels"], names, pos_idx)
            y_tst = _labels_from_cache(cached["test_labels"], names, pos_idx)
        est_list.append(_probs_from_cache(cached["est_probs"], names, pos_idx))
        cal_list.append(_probs_from_cache(cached["cal_probs"], names, pos_idx))
        tst_list.append(_probs_from_cache(cached["test_probs"], names, pos_idx))
    assert names is not None and y_est is not None and y_cal is not None and y_tst is not None
    return (
        np.stack(est_list, axis=1),
        np.stack(cal_list, axis=1),
        np.stack(tst_list, axis=1),
        y_est,
        y_cal,
        y_tst,
        names,
    )


def _compute_preds_at_stop(pos_scores, stop_indices, b_pos, b_neg) -> np.ndarray:
    n = pos_scores.shape[0]
    scores_at_stop = pos_scores[np.arange(n), stop_indices, :]
    b_pos_t = b_pos[stop_indices]
    b_neg_t = b_neg[stop_indices]
    center = b_neg_t / (1.0 + b_neg_t - b_pos_t)
    return (scores_at_stop >= center).astype(int)


def _rcs_curves_for_lambda(
    test_pos: np.ndarray,
    labels: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    lambda_hat: float,
    control_idxs: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, Any]:
    from fst.ambiguity_bands import compute_bounds, compute_stop_indices

    lower, upper = compute_bounds(float(lambda_hat), b_pos, b_neg)
    stop_indices, stopped = compute_stop_indices(test_pos, lower, upper)
    preds = _compute_preds_at_stop(test_pos, stop_indices, b_pos, b_neg)
    ctrl_names = [label_names[i] for i in control_idxs]
    t_steps = test_pos.shape[1]
    out: Dict[str, Any] = {
        "stopped_count": [],
        "fnr_cumulative": [],
        "balanced_accuracy_cumulative": [],
        "fnr_cumulative_per_label": {n: [] for n in ctrl_names},
    }
    for t in range(t_steps):
        idx = np.where(stopped & (stop_indices <= t))[0]
        out["stopped_count"].append(int(idx.size))
        fnrs = [_fnr_on_idx(preds, labels, idx, int(i)) for i in control_idxs]
        out["fnr_cumulative"].append(float(max(fnrs)) if fnrs else 0.0)
        out["balanced_accuracy_cumulative"].append(_ba_on_idx(preds, labels, idx))
        for i, name in zip(control_idxs, ctrl_names):
            out["fnr_cumulative_per_label"][name].append(_fnr_on_idx(preds, labels, idx, int(i)))
    return out


def _select_lambda_hat(
    cal_pos: np.ndarray,
    y_cal: np.ndarray,
    b_pos: np.ndarray,
    b_neg: np.ndarray,
    *,
    alpha_per_label: np.ndarray,
    delta: float,
    control_idxs: Sequence[int],
    num_lambdas: int = 15,
    utility: str = "stopped_count",
) -> Tuple[Optional[float], np.ndarray]:
    from fst.ambiguity_bands import compute_bounds, compute_stop_indices
    from fst.ltt import grid_class_conditional_pvalues, grid_utilities, ltt_select_lambda

    lambdas = np.linspace(0.0, 1.0, int(num_lambdas))
    n_cal = y_cal.shape[0]
    m = lambdas.shape[0]
    k = y_cal.shape[1]
    stopped_mat = np.zeros((n_cal, m), dtype=bool)
    stop_idx_mat = np.zeros((n_cal, m), dtype=int)
    preds_cube = np.zeros((n_cal, m, k), dtype=int)
    for j, lam in enumerate(lambdas):
        lower, upper = compute_bounds(float(lam), b_pos, b_neg)
        stop_indices, stopped = compute_stop_indices(cal_pos, lower, upper)
        preds_cube[:, j, :] = _compute_preds_at_stop(cal_pos, stop_indices, b_pos, b_neg)
        stopped_mat[:, j] = stopped
        stop_idx_mat[:, j] = stop_indices

    p_cond, risks_cond, _ns, _details = grid_class_conditional_pvalues(
        preds_cube,
        y_cal,
        stopped_mat,
        alpha_per_label,
        risk_metric="fnr",
        pvalue_method="binomial",
        control_both=False,
        control_label_indices=list(control_idxs),
    )
    utilities = grid_utilities(stopped_mat, stop_idx_mat, risks_cond, utility)
    lam_hat, meta = ltt_select_lambda(lambdas, risks_cond, p_cond, float(delta), utilities)
    print(
        f"LTT α_k={alpha_per_label.tolist()} control={list(control_idxs)} "
        f"→ λ̂={lam_hat} reason={meta.get('reason')}",
        flush=True,
    )
    return lam_hat, lambdas


def _load_test_cube(score_cache_dir: Path, sampling_rates: Sequence[float]):
    _est, _cal, tst, _ye, _yc, yt, names = _load_score_cubes(score_cache_dir, sampling_rates)
    return tst, yt, names


def _draw_stopped(ax, rates, n_test, series: List[Tuple[str, str, Sequence[float]]]):
    inv = 1.0 / float(n_test)
    for key, lab, counts in series:
        frac = [int(c) * inv for c in counts]
        _plot(ax, rates, frac, key=key, label=lab, zorder=3 if key == "ltt_conditional" else 2)
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel("Fraction stopped")
    ax.set_ylim(0.0, 1.05)
    _style_ax(ax)


def _draw_risk(ax, rates, series, alpha: float, ylabel: str):
    draw_alpha_reference(ax, float(alpha))
    primary_ys = []
    forced_ys = None
    for key, lab, vals in series:
        _plot(ax, rates, vals, key=key, label=lab, zorder=3 if key == "ltt_conditional" else 2)
        if key == "forced":
            forced_ys = vals
        else:
            primary_ys.append(vals)
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel(ylabel)
    ymin, ymax = _panel_ylim(primary_ys, forced_ys, alpha)
    ax.set_ylim(ymin, ymax)
    _style_ax(ax)


def _draw_ba(ax, rates, series):
    for key, lab, vals in series:
        _plot(ax, rates, vals, key=key, label=lab, zorder=3 if key == "ltt_conditional" else 2)
    ax.set_xlabel(COMPARE_XLABEL_SAMPLING)
    ax.set_ylabel("Cumulative balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    _style_ax(ax)


def _series_bundle(
    curves: Dict[str, Any],
    *,
    conf: Optional[Dict[str, Any]],
    ent: Optional[Dict[str, Any]],
    conf_thr: Optional[float],
    ent_thr: Optional[float],
    metric: str,
    label_name: Optional[str] = None,
) -> List[Tuple[str, str, Sequence[float]]]:
    def pick(curve: Dict[str, Any]) -> Sequence[float]:
        if label_name is not None and metric.startswith("fnr") and "fnr_cumulative_per_label" in curve:
            return curve["fnr_cumulative_per_label"][label_name]
        return curve[metric]

    out = [
        ("ltt_conditional", COMPARE_RCS_LEGEND_LABEL, pick(curves["ltt_conditional"])),
        ("forced", _forced_label(), pick(curves["forced"])),
    ]
    if curves.get("cal_tuned_forced") is not None:
        out.append(("cal_tuned_forced", _cal_tuned_label(), pick(curves["cal_tuned_forced"])))
    if conf is not None and conf_thr is not None:
        out.append(("confidence_threshold", _conf_label(conf_thr), pick(conf)))
    if ent is not None and ent_thr is not None:
        out.append(("entropy_threshold", _ent_label(ent_thr), pick(ent)))
    return out


def plot_knee_2x2(
    out_dir: Path,
    rates: List[float],
    n_test: int,
    curves: Dict[str, Any],
    conf: Dict[str, Any],
    ent: Dict[str, Any],
    conf_thr: float,
    ent_thr: float,
    ctrl_names: List[str],
    alpha_by_label: Dict[str, float],
) -> Path:
    fig, axes = plt.subplots(
        2, 2, figsize=COMPARE_FIGSIZE_2x2, dpi=300, sharex=True, constrained_layout=False
    )
    ax0, ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    stop_series = [
        ("ltt_conditional", COMPARE_RCS_LEGEND_LABEL, curves["ltt_conditional"]["stopped_count"]),
        ("forced", _forced_label(), curves["forced"]["stopped_count"]),
        ("cal_tuned_forced", _cal_tuned_label(), curves["cal_tuned_forced"]["stopped_count"]),
        ("confidence_threshold", _conf_label(conf_thr), conf["stopped_count"]),
        ("entropy_threshold", _ent_label(ent_thr), ent["stopped_count"]),
    ]
    _draw_stopped(ax0, rates, n_test, stop_series)

    name0, name1 = ctrl_names[0], ctrl_names[1]
    _draw_risk(
        ax1,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="fnr_cumulative", label_name=name0),
        alpha_by_label[name0],
        f"Stopped FNR ({_label(name0)})",
    )
    _draw_risk(
        ax2,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="fnr_cumulative", label_name=name1),
        alpha_by_label[name1],
        f"Stopped FNR ({_label(name1)})",
    )
    _draw_ba(
        ax3,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="balanced_accuracy_cumulative"),
    )

    prio = [
        COMPARE_RCS_LEGEND_LABEL,
        _forced_label(),
        _cal_tuned_label(),
        _conf_label(conf_thr),
        _ent_label(ent_thr),
    ]
    for a in sorted(set(alpha_by_label.values())):
        prio.append(f"α = {float(a):g}")
    handles, labels = _merge_legend([ax0, ax1, ax2, ax3], prio)
    top = FASTMRI_PANEL_LAYOUT_TOP
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, top),
        ncol=len(handles),  # force a single horizontal row
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.82,
        borderaxespad=0.35,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.55,
        handlelength=2.0,
    )
    out = out_dir / "compare_panel_knee_slice_fnr_conditional.png"
    save_compare_figure(fig, out, dpi=300)
    return out


def plot_knee_1x3_no_ba(
    out_dir: Path,
    rates: List[float],
    n_test: int,
    curves: Dict[str, Any],
    conf: Dict[str, Any],
    ent: Dict[str, Any],
    conf_thr: float,
    ent_thr: float,
    ctrl_names: List[str],
    alpha_by_label: Dict[str, float],
) -> Path:
    """
    Compact 1×3 knee panel: fraction stopped + per-label stopped FNR (no balanced accuracy).

    Writes a separate file so the existing 2×2 panel is left unchanged.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=COMPARE_FIGSIZE_KNEE_1x3, dpi=300, sharex=True, constrained_layout=False
    )
    ax0, ax1, ax2 = axes

    stop_series = [
        ("ltt_conditional", COMPARE_RCS_LEGEND_LABEL, curves["ltt_conditional"]["stopped_count"]),
        ("forced", _forced_label(), curves["forced"]["stopped_count"]),
        ("cal_tuned_forced", _cal_tuned_label(), curves["cal_tuned_forced"]["stopped_count"]),
        ("confidence_threshold", _conf_label(conf_thr), conf["stopped_count"]),
        ("entropy_threshold", _ent_label(ent_thr), ent["stopped_count"]),
    ]
    _draw_stopped(ax0, rates, n_test, stop_series)

    name0, name1 = ctrl_names[0], ctrl_names[1]
    _draw_risk(
        ax1,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="fnr_cumulative", label_name=name0),
        alpha_by_label[name0],
        f"Stopped FNR ({_label(name0)})",
    )
    _draw_risk(
        ax2,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="fnr_cumulative", label_name=name1),
        alpha_by_label[name1],
        f"Stopped FNR ({_label(name1)})",
    )

    prio = [
        COMPARE_RCS_LEGEND_LABEL,
        _forced_label(),
        _cal_tuned_label(),
        _conf_label(conf_thr),
        _ent_label(ent_thr),
    ]
    for a in sorted(set(alpha_by_label.values())):
        prio.append(f"α = {float(a):g}")
    handles, labels = _merge_legend([ax0, ax1, ax2], prio)
    top = 0.84
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, top),
        ncol=len(handles),
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.78,
        borderaxespad=0.3,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.5,
        handlelength=1.9,
    )
    out = out_dir / "compare_panel_knee_slice_fnr_conditional_1x3.png"
    save_compare_figure(fig, out, dpi=300)
    return out


def plot_prostate_1x3(
    out_dir: Path,
    rates: List[float],
    n_test: int,
    curves: Dict[str, Any],
    conf: Optional[Dict[str, Any]],
    ent: Optional[Dict[str, Any]],
    conf_thr: Optional[float],
    ent_thr: Optional[float],
    alpha: float,
    file_tag: str,
) -> Path:
    fig, axes = plt.subplots(
        1, 3, figsize=COMPARE_FIGSIZE_3COL, dpi=300, sharex=True, constrained_layout=False
    )
    ax0, ax1, ax2 = axes

    stop_series = [
        ("ltt_conditional", COMPARE_RCS_LEGEND_LABEL, curves["ltt_conditional"]["stopped_count"]),
        ("forced", _forced_label(), curves["forced"]["stopped_count"]),
    ]
    if curves.get("cal_tuned_forced") is not None:
        stop_series.append(
            ("cal_tuned_forced", _cal_tuned_label(), curves["cal_tuned_forced"]["stopped_count"])
        )
    if conf is not None and conf_thr is not None:
        stop_series.append(("confidence_threshold", _conf_label(conf_thr), conf["stopped_count"]))
    if ent is not None and ent_thr is not None:
        stop_series.append(("entropy_threshold", _ent_label(ent_thr), ent["stopped_count"]))
    _draw_stopped(ax0, rates, n_test, stop_series)

    _draw_risk(
        ax1,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="fnr_cumulative"),
        alpha,
        f"Stopped FNR ({_label('pirads_positive')})",
    )
    _draw_ba(
        ax2,
        rates,
        _series_bundle(curves, conf=conf, ent=ent, conf_thr=conf_thr, ent_thr=ent_thr,
                       metric="balanced_accuracy_cumulative"),
    )

    prio = [COMPARE_RCS_LEGEND_LABEL, _forced_label(), _cal_tuned_label()]
    if conf_thr is not None:
        prio.append(_conf_label(conf_thr))
    if ent_thr is not None:
        prio.append(_ent_label(ent_thr))
    prio.append(f"α = {float(alpha):g}")
    handles, labels = _merge_legend([ax0, ax1, ax2], prio)
    top = 0.86
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, top),
        ncol=len(handles),  # force a single horizontal row
        fontsize=COMPARE_LEGEND_FONTSIZE * 0.92,
        borderaxespad=0.4,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.55,
        handlelength=2.1,
    )
    out = out_dir / f"compare_panel_{file_tag}_conditional.png"
    save_compare_figure(fig, out, dpi=300)
    return out


def replot_knee(
    summary_path: Path,
    score_cache_dir: Path,
    out_dir: Path,
    conf_thr: float,
    ent_thr: float,
    *,
    alpha: float = 0.1,
    num_lambdas: int = 15,
    num_deltas: int = 101,
):
    from fst.ambiguity_bands import compute_bands

    summary = json.loads(summary_path.read_text())
    curves = dict(summary["curves"])
    rates = [float(x) for x in curves["sampling_rates"]]
    n_test = int(summary["n_test"])
    est_pos, cal_pos, test_pos, y_est, y_cal, y_tst, names = _load_score_cubes(
        score_cache_dir, rates
    )
    if test_pos.shape[0] != n_test:
        raise ValueError(f"n_test mismatch cache={test_pos.shape[0]} summary={n_test}")

    exclude = summary.get("exclude_labels") or "abnormal"
    excl = {s.strip() for s in str(exclude).split(",") if s.strip()}
    control_idxs = [i for i, n in enumerate(names) if n not in excl]
    ctrl_names = [names[i] for i in control_idxs]
    # Control every selected pathology at the same α.
    alpha_k = np.full(len(names), float(alpha), dtype=float)
    alpha_by = {n: float(alpha) for n in ctrl_names}
    delta = float(summary.get("delta", 0.1))

    b_pos, b_neg, _ = compute_bands(est_pos, y_est, int(num_deltas))
    lam_hat, _ = _select_lambda_hat(
        cal_pos,
        y_cal,
        b_pos,
        b_neg,
        alpha_per_label=alpha_k,
        delta=delta,
        control_idxs=control_idxs,
        num_lambdas=int(num_lambdas),
        utility=str(summary.get("ltt_conditional_utility", "stopped_count")),
    )
    if lam_hat is None:
        raise RuntimeError("LTT selected no safe λ for the requested α")
    curves["ltt_conditional"] = _rcs_curves_for_lambda(
        test_pos, y_tst, b_pos, b_neg, float(lam_hat), control_idxs, names
    )

    conf = _heuristic_curves(
        test_pos, y_tst, mode="confidence", threshold=conf_thr,
        control_idxs=control_idxs, label_names=names,
    )
    ent = _heuristic_curves(
        test_pos, y_tst, mode="entropy", threshold=ent_thr,
        control_idxs=control_idxs, label_names=names,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = plot_knee_2x2(
        out_dir, rates, n_test, curves, conf, ent, conf_thr, ent_thr, ctrl_names, alpha_by
    )

    # Persist an updated summary next to the panel for reproducibility.
    summary_out = dict(summary)
    summary_out["alpha"] = float(alpha)
    summary_out["alpha_per_label"] = alpha_k.tolist()
    summary_out["exclude_labels"] = ",".join(sorted(excl)) if excl else None
    summary_out["control_label_names"] = ctrl_names
    summary_out["confidence_threshold"] = float(conf_thr)
    summary_out["entropy_threshold"] = float(ent_thr)
    summary_out["selections"] = {
        **summary.get("selections", {}),
        "ltt_conditional": {
            "lambda_hat": float(lam_hat),
            "reason": "ltt_selected",
            "conditional_test": "class_conditional_binomial",
        },
    }
    summary_out["curves"] = {
        **curves,
        "confidence_threshold_baseline": conf,
        "entropy_threshold_baseline": ent,
    }
    out_json = out_dir / "compare_knee_summary_knee_slice_fnr.json"
    out_json.write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(f"Wrote {out_json}", flush=True)


def replot_prostate(summary_path: Path, out_dir: Path, file_tag: str):
    summary = json.loads(summary_path.read_text())
    curves = summary["curves"]
    rates = [float(x) for x in curves["sampling_rates"]]
    n_test = int(summary["n_test"])
    alpha = float(summary["alpha"])
    conf = curves.get("confidence_threshold_baseline")
    ent = curves.get("entropy_threshold_baseline")
    conf_thr = summary.get("confidence_threshold")
    ent_thr = summary.get("entropy_threshold")
    if conf is None or ent is None:
        print("WARNING: summary missing Confidence/Entropy curves.", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = plot_prostate_1x3(
        out_dir, rates, n_test, curves, conf, ent, conf_thr, ent_thr, alpha, file_tag
    )
    print(f"Wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pk = sub.add_parser("knee")
    pk.add_argument("--summary", required=True)
    pk.add_argument("--score_cache_dir", required=True)
    pk.add_argument("--out_dir", required=True)
    pk.add_argument("--confidence_threshold", type=float, default=0.8)
    pk.add_argument("--entropy_threshold", type=float, default=0.2)
    pk.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Class-conditional FNR level applied to every controlled pathology (default: 0.1).",
    )
    pk.add_argument("--num_lambdas", type=int, default=15)
    pk.add_argument("--num_deltas", type=int, default=101)
    pp = sub.add_parser("prostate")
    pp.add_argument("--summary", required=True)
    pp.add_argument("--out_dir", required=True)
    pp.add_argument("--file_tag", default="prostate_pirads_fnr")
    args = p.parse_args()
    configure_compare_plot_style()
    if args.cmd == "knee":
        replot_knee(
            Path(args.summary), Path(args.score_cache_dir), Path(args.out_dir),
            float(args.confidence_threshold), float(args.entropy_threshold),
            alpha=float(args.alpha),
            num_lambdas=int(args.num_lambdas),
            num_deltas=int(args.num_deltas),
        )
    else:
        replot_prostate(Path(args.summary), Path(args.out_dir), str(args.file_tag))


if __name__ == "__main__":
    main()
