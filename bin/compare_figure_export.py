#!/usr/bin/env python3
"""Shared matplotlib export for eval_compare_* scripts: PNG + vector PDF for papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt

# Rasterization fallback for PDF when elements are not pure vector; standard print resolution.
COMPARE_PUBLICATION_DPI = 300

# Softer grid than default compare scripts used (alpha=0.3); tune here for all panels.
COMPARE_GRID_ALPHA = 0.22

# X-axis for acquisition / sampling curves (values are fractions in [0, 1], not percents).
COMPARE_XLABEL_SAMPLING = "Sampling rate"

# Panel face / spine colors shared by compare + α-sweep figures.
COMPARE_AX_FACE = "#F7F8FA"
COMPARE_SPINE_COLOR = "#9AA3AD"
COMPARE_TICK_COLOR = "#4A5560"
COMPARE_LABEL_COLOR = "#2F3A44"
COMPARE_GRID_COLOR = "#D5DBE1"
# Horizontal α control reference on FNR/FPR panels.
# Solid soft-rose line so it reads as a threshold, not another method curve.
COMPARE_ALPHA_LINE_COLOR = "#D08A8C"
COMPARE_ALPHA_LINE_WIDTH = 2.25
COMPARE_ALPHA_LINE_STYLE = "-"
COMPARE_ALPHA_LINE_ALPHA = 0.90
COMPARE_ALPHA_LINE_ZORDER = 2.5


# Typography targets (matplotlib points; readable at typical LaTeX figure widths).
COMPARE_FONT_SIZE = 22.0
COMPARE_AXIS_LABELSIZE = 22.0
COMPARE_TICK_LABELSIZE = 16.0
COMPARE_TITLE_SIZE = 22.0
# Figure-level legends must pass ``fontsize=...`` explicitly — rcParams alone is easy to miss.
COMPARE_LEGEND_FONTSIZE = 22.0

# Short name for ``ltt_conditional`` curves. Matplotlib mathtext does not support ``\\textsc``;
# Unicode small caps match the intended ``$\\textsc{rcs}$`` look without enabling usetex.
COMPARE_RCS_LEGEND_LABEL = (
    "\N{LATIN LETTER SMALL CAPITAL R}"
    "\N{LATIN LETTER SMALL CAPITAL C}"
    "\N{LATIN LETTER SMALL CAPITAL S}"
)

# Figure sizes (inches) — large canvas so 16–18 pt text is not cramped.
# 1×3: between the old ultra-wide short strip (30×7.35) and a tall portrait
# layout (20.5×9.2), so panels stay landscape while text still scales near the 2×2.
COMPARE_FIGSIZE_SINGLE = (11.0, 6.75)
COMPARE_FIGSIZE_2COL = (20.0, 6.95)
COMPARE_FIGSIZE_3COL = (25.0, 8.0)
COMPARE_FIGSIZE_2x2 = (14.0, 10.5)
# Compact knee strip: fraction stopped + two FNR panels (no balanced accuracy).
COMPARE_FIGSIZE_KNEE_1x3 = (16.5, 5.5)


def configure_compare_plot_style(**overrides: Any) -> None:
    """
    Publication-oriented type sizes for compare figures (readable when scaled in LaTeX).

    Call once near the start of each ``eval_compare_*`` ``main()`` (before building figures).
    """
    cfg: dict[str, Any] = {
        "font.size": COMPARE_FONT_SIZE,
        "axes.labelsize": COMPARE_AXIS_LABELSIZE,
        "axes.titlesize": COMPARE_TITLE_SIZE,
        "xtick.labelsize": COMPARE_TICK_LABELSIZE,
        "ytick.labelsize": COMPARE_TICK_LABELSIZE,
        "legend.fontsize": COMPARE_LEGEND_FONTSIZE,
        "figure.titlesize": COMPARE_TITLE_SIZE,
        "axes.linewidth": 0.9,
        "axes.edgecolor": COMPARE_SPINE_COLOR,
        "axes.facecolor": COMPARE_AX_FACE,
        "figure.facecolor": "white",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.color": COMPARE_TICK_COLOR,
        "ytick.color": COMPARE_TICK_COLOR,
        "axes.labelcolor": COMPARE_LABEL_COLOR,
        "axes.titlecolor": "#1F2933",
        "lines.linewidth": 2.35,
        "lines.markersize": 8.5,
        "grid.color": COMPARE_GRID_COLOR,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "legend.frameon": False,
        "legend.fancybox": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E6EA",
        "legend.handlelength": 2.35,
        "legend.labelspacing": 0.95,
        "legend.columnspacing": 1.75,
        "legend.handletextpad": 0.85,
    }
    cfg.update(overrides)
    plt.rcParams.update(cfg)


def style_compare_ax(ax: Any, *, grid: bool = True, grid_y: bool = True) -> None:
    """
    Shared publication axis styling (despine, soft face, light grid).

    Used by FastMRI / prostate compare panels and α-sweep figures.
    ``grid_y`` is kept for older call sites that only wanted horizontal lines.
    """
    ax.set_facecolor(COMPARE_AX_FACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COMPARE_SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.9)
    if grid:
        ax.grid(True, which="major", axis="both", color=COMPARE_GRID_COLOR, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
    elif grid_y:
        ax.grid(True, axis="y", alpha=COMPARE_GRID_ALPHA, linewidth=0.55, color=COMPARE_GRID_COLOR)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.2, width=0.7, colors=COMPARE_TICK_COLOR)
    ax.xaxis.label.set_color(COMPARE_LABEL_COLOR)
    ax.yaxis.label.set_color(COMPARE_LABEL_COLOR)
    if ax.title.get_text():
        ax.title.set_color("#1F2933")


def draw_alpha_reference(ax: Any, control_alpha: float, *, label: Optional[str] = None) -> None:
    """Draw a solid α threshold line (distinct from method curves)."""
    ca = float(control_alpha)
    ax.axhline(
        ca,
        linestyle=COMPARE_ALPHA_LINE_STYLE,
        linewidth=COMPARE_ALPHA_LINE_WIDTH,
        color=COMPARE_ALPHA_LINE_COLOR,
        alpha=COMPARE_ALPHA_LINE_ALPHA,
        zorder=COMPARE_ALPHA_LINE_ZORDER,
        solid_capstyle="round",
        label=label if label is not None else f"α = {ca:g}",
    )


def draw_accuracy_target_reference(
    ax: Any, target_acc: float, *, label: Optional[str] = None
) -> None:
    """Solid 1−α accuracy target (same visual language as ``draw_alpha_reference``)."""
    ta = float(target_acc)
    ax.axhline(
        ta,
        linestyle=COMPARE_ALPHA_LINE_STYLE,
        linewidth=COMPARE_ALPHA_LINE_WIDTH,
        color=COMPARE_ALPHA_LINE_COLOR,
        alpha=COMPARE_ALPHA_LINE_ALPHA,
        zorder=COMPARE_ALPHA_LINE_ZORDER,
        solid_capstyle="round",
        label=label if label is not None else f"1−α = {ta:g}",
    )


def plot_compare_series(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
    zorder: int = 2,
) -> None:
    """
    Shared method-curve style for MRI / CIFAR / JAFA panels.

    Filled markers get a white halo; edge-only markers (``x``, ``+``, …) keep the
    series color so points stay visible.
    """
    edge_only = str(marker) in {"x", "X", "+", "1", "2", "3", "4", "|", "_"}
    ax.plot(
        list(xs),
        list(ys),
        color=color,
        linestyle=linestyle,
        marker=marker,
        label=label,
        markeredgecolor=color if edge_only else "white",
        markeredgewidth=1.15 if edge_only else 0.85,
        markersize=8.0 if edge_only else 7.5,
        linewidth=2.35,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )


def print_label_prevalence(
    rows: Sequence[tuple[str, int, int]],
    *,
    split_name: str = "test",
    controlled_names: Optional[Sequence[str]] = None,
) -> None:
    """
    Print per-label positive prevalence before the plot-data table.

    ``rows`` is ``(label_name, n_positive, n_total)`` for each label.
    """
    controlled = {str(n) for n in controlled_names} if controlled_names else None
    print("", flush=True)
    print(f"Label prevalence ({split_name}):", flush=True)
    for name, n_pos, n_total in rows:
        n_pos_i = int(n_pos)
        n_tot_i = int(n_total)
        rate = (float(n_pos_i) / float(n_tot_i)) if n_tot_i > 0 else 0.0
        flag = ""
        if controlled is not None:
            flag = "  [controlled]" if str(name) in controlled else "  [excluded from control]"
        print(
            f"  {name}: {rate:.4f}  ({n_pos_i}/{n_tot_i}){flag}",
            flush=True,
        )
    print("", flush=True)


def _curve_metric_series(curve: Mapping[str, Any], key: str) -> Optional[Sequence[Any]]:
    """
    Resolve a metric series from a curve dict.

    Supports flat keys (``fnr_cumulative``) and nested per-label keys
    (``fnr_cumulative_per_label.acl`` → ``curve["fnr_cumulative_per_label"]["acl"]``).
    """
    if key in curve:
        return curve.get(key)  # type: ignore[return-value]
    if "." in key:
        head, tail = key.split(".", 1)
        nested = curve.get(head)
        if isinstance(nested, Mapping):
            return nested.get(tail)  # type: ignore[return-value]
    return None


def print_compare_curve_table(
    sampling_rates: Sequence[float],
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    metric_key: str | Sequence[str],
    count_key: Optional[str] = "stopped_count",
    title: Optional[str] = None,
    metric_label: Optional[str | Sequence[str]] = None,
    metric_digits: int = 4,
    metric_count_key: Optional[str | Sequence[str]] = None,
) -> None:
    """
    Print a methods × sampling-rate table of plot-aligned cells.

    ``rows`` is an ordered list of ``(legend_label, curve_dict)`` where each
    ``curve_dict`` has the metric series (and ``count_key`` when set) aligned
    with ``sampling_rates``.

    ``metric_key`` may be a single key or a sequence of keys (including nested
    ``per_label`` keys like ``fnr_cumulative_per_label.acl``).

    Optional ``metric_count_key`` (parallel to ``metric_key``) supplies
    ``(n_err, n_denom)`` pairs shown beside each rate as ``rate (n_err/n_denom)``.

    With ``count_key`` set (default), cells are ``(stopped_count, metric...)``.
    With ``count_key=None``, cells are the metric value(s) alone.
    """
    rates = [float(r) for r in sampling_rates]
    n_rates = len(rates)
    if n_rates == 0:
        return

    metric_keys: list[str] = (
        [str(metric_key)]
        if isinstance(metric_key, str)
        else [str(k) for k in metric_key]
    )
    if metric_count_key is None:
        metric_count_keys: list[Optional[str]] = [None] * len(metric_keys)
    elif isinstance(metric_count_key, str):
        metric_count_keys = [metric_count_key]
    else:
        metric_count_keys = [str(k) for k in metric_count_key]
    if len(metric_count_keys) != len(metric_keys):
        raise ValueError("metric_count_key length must match metric_key length")

    if isinstance(metric_label, str) or metric_label is None:
        if metric_label is not None:
            metric_labels = [str(metric_label)]
        else:
            metric_labels = list(metric_keys)
        if len(metric_labels) == 1 and len(metric_keys) > 1:
            metric_labels = list(metric_keys)
    else:
        metric_labels = [str(x) for x in metric_label]
        if len(metric_labels) != len(metric_keys):
            raise ValueError("metric_label length must match metric_key length")

    metrics_desc = ", ".join(metric_labels)
    if title is None:
        if count_key is None:
            hdr_title = f"Plot data table  (cell = {metrics_desc})"
        else:
            hdr_title = f"Plot data table  (cell = (stopped_count, {metrics_desc}))"
    else:
        hdr_title = title
    col_headers = [f"r={r:g}" for r in rates]

    def _fmt_metric(raw: Any, count_raw: Any = None) -> str:
        try:
            rate_s = f"{float(raw):.{metric_digits}f}"
        except (TypeError, ValueError):
            rate_s = str(raw)
        if count_raw is None:
            return rate_s
        if isinstance(count_raw, (tuple, list)) and len(count_raw) >= 2:
            try:
                n_err = int(count_raw[0])
                n_den = int(count_raw[1])
            except (TypeError, ValueError):
                return f"{rate_s} ({count_raw[0]}/{count_raw[1]})"
            return f"{rate_s} ({n_err}/{n_den})"
        return rate_s

    # Resolve cells; skip rows missing any requested metric.
    resolved: list[tuple[str, list[str]]] = []
    for label, curve in rows:
        if curve is None:
            continue
        series_list = [_curve_metric_series(curve, mk) for mk in metric_keys]
        if any(s is None for s in series_list):
            continue
        count_series_list: list[Optional[Sequence[Any]]] = []
        for ck in metric_count_keys:
            if ck is None:
                count_series_list.append(None)
            else:
                count_series_list.append(_curve_metric_series(curve, ck))
        # If any count key was requested but missing, skip count decoration for that metric.
        counts = None if count_key is None else curve.get(count_key)
        if count_key is not None and counts is None:
            continue
        cells: list[str] = []
        for j in range(n_rates):
            if any(j >= len(s) for s in series_list):  # type: ignore[arg-type]
                cells.append("—")
                continue
            if counts is not None and j >= len(counts):
                cells.append("—")
                continue
            m_parts: list[str] = []
            for s, cs in zip(series_list, count_series_list):
                c_raw = None
                if cs is not None and j < len(cs):
                    c_raw = cs[j]
                m_parts.append(_fmt_metric(s[j], c_raw))  # type: ignore[index]
            if counts is None:
                cells.append(m_parts[0] if len(m_parts) == 1 else f"({', '.join(m_parts)})")
            else:
                c_raw = counts[j]
                try:
                    c_i = int(c_raw)
                except (TypeError, ValueError):
                    c_i = c_raw
                cells.append(f"({c_i}, {', '.join(m_parts)})")
        resolved.append((str(label), cells))

    if not resolved:
        return

    method_w = max(len("method"), max(len(lab) for lab, _ in resolved))
    col_w = [
        max(len(col_headers[j]), max(len(cells[j]) for _, cells in resolved))
        for j in range(n_rates)
    ]

    def _fmt_row(method: str, cells: Sequence[str]) -> str:
        parts = [f"{method:<{method_w}}"]
        for j, cell in enumerate(cells):
            parts.append(f"{cell:>{col_w[j]}}")
        return "  ".join(parts)

    print("", flush=True)
    print(hdr_title, flush=True)
    print(_fmt_row("method", col_headers), flush=True)
    print(
        _fmt_row("-" * method_w, ["-" * w for w in col_w]),
        flush=True,
    )
    for lab, cells in resolved:
        print(_fmt_row(lab, cells), flush=True)
    print("", flush=True)


def save_compare_figure(
    fig: Any,
    out_path: Path | str,
    *,
    dpi: int = COMPARE_PUBLICATION_DPI,
    close: bool = True,
    **kwargs: Any,
) -> None:
    """
    Write ``out_path`` (typically ``*.png``) and the same basename with ``.pdf``.

    PDFs are suitable for LaTeX/publication inclusion (vector graphics; dpi affects embedded
    raster elements only).
    """
    out_path = Path(out_path)
    save_kw: dict[str, Any] = {"bbox_inches": "tight", "dpi": dpi, **kwargs}
    fig.savefig(out_path, **save_kw)

    pdf_path = out_path.with_suffix(".pdf")
    pdf_kw = dict(save_kw)
    pdf_kw["format"] = "pdf"
    fig.savefig(pdf_path, **pdf_kw)

    print(f"Saved plot -> {out_path}")
    print(f"Saved plot -> {pdf_path}")

    if close:
        plt.close(fig)
