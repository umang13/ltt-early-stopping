"""Helpers for comparing saved image-model prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.analyze_image_checkpoints import build_context, load_base_args
from prostate_kspace_classifier.data.h5_dataset import create_image_dataset


PREDICTION_FILE_NAMES = {
    "val": {
        "ensemble": ("val_ensemble_predictions", "val_ensemble_predictions.csv"),
        "seed": ("val_seed_predictions", "val_seed_predictions.csv"),
    },
    "test": {
        "ensemble": ("test_ensemble_predictions", "test_ensemble_predictions.csv"),
        "seed": ("test_seed_predictions", "test_seed_predictions.csv"),
        "legacy_ensemble": ("ensemble_predictions", "ensemble_predictions.csv"),
        "legacy_seed": ("seed_predictions", "seed_predictions.csv"),
    },
}


@dataclass(frozen=True)
class AnalysisRun:
    repo_root: Path
    analysis_dir: Path
    summary: dict[str, Any]
    config_path: Path
    experiment_name: str | None
    base_args: dict[str, Any]

    @property
    def run_metadata(self) -> dict[str, Any]:
        return dict(self.summary.get("run_metadata", {}))

    @property
    def checkpoint_count(self) -> int:
        return int(self.summary.get("checkpoint_count", len(self.summary.get("checkpoints", []))))


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "scripts").exists():
            return candidate.resolve()
    raise RuntimeError("Could not find repo root from the current working directory.")


def resolve_repo_path(repo_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (repo_root / path)


def resolve_summary_config_path(
    repo_root: Path,
    summary: dict[str, Any],
) -> Path:
    config_path = resolve_repo_path(repo_root, summary["config_path"])
    if config_path.exists():
        return config_path

    basename = Path(summary["config_path"]).name
    candidates = sorted((repo_root / "configs").glob(f"**/{basename}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve config path '{summary['config_path']}' from {repo_root}"
        )
    if len(candidates) == 1:
        return candidates[0].resolve()

    checkpoint_text = " ".join(str(path) for path in summary.get("checkpoints", []))
    inferred_coil = None
    if "-espirit-" in checkpoint_text:
        inferred_coil = "espirit"
    elif "-rss-" in checkpoint_text:
        inferred_coil = "rss"

    if inferred_coil is not None:
        filtered = [
            candidate
            for candidate in candidates
            if inferred_coil in candidate.parts or inferred_coil in str(candidate)
        ]
        if len(filtered) == 1:
            return filtered[0].resolve()

    raise FileNotFoundError(
        "Could not uniquely resolve config path "
        f"'{summary['config_path']}'. Candidates: {[str(path) for path in candidates]}"
    )


def load_analysis_run(analysis_dir: str | Path, *, repo_root: Path | None = None) -> AnalysisRun:
    repo_root = (repo_root or find_repo_root(Path.cwd())).resolve()
    analysis_path = resolve_repo_path(repo_root, analysis_dir)
    summary_path = analysis_path / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    config_path = resolve_summary_config_path(repo_root, summary)
    experiment_name = summary.get("experiment_name")
    base_args = load_base_args(config_path, experiment_name)
    return AnalysisRun(
        repo_root=repo_root,
        analysis_dir=analysis_path,
        summary=summary,
        config_path=config_path,
        experiment_name=experiment_name,
        base_args=base_args,
    )


def _resolve_prediction_path(run: AnalysisRun, split: str, kind: str) -> Path | None:
    outputs = run.summary.get("outputs", {})
    mapping = PREDICTION_FILE_NAMES[split]
    key_name, default_name = mapping[kind]
    candidate = outputs.get(key_name)
    if candidate:
        return resolve_repo_path(run.repo_root, candidate)
    fallback = run.analysis_dir / default_name
    if fallback.exists():
        return fallback

    legacy_key = f"legacy_{kind}"
    if legacy_key in mapping:
        legacy_key_name, legacy_default_name = mapping[legacy_key]
        legacy_candidate = outputs.get(legacy_key_name)
        if legacy_candidate:
            return resolve_repo_path(run.repo_root, legacy_candidate)
        legacy_fallback = run.analysis_dir / legacy_default_name
        if legacy_fallback.exists():
            return legacy_fallback

    return None


def load_prediction_table(
    run: AnalysisRun,
    *,
    split: str = "test",
    prediction_mode: str = "auto",
    single_checkpoint_index: int = 0,
) -> pd.DataFrame:
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split '{split}'")
    if prediction_mode not in {"auto", "ensemble", "single"}:
        raise ValueError(
            f"prediction_mode must be one of auto/ensemble/single, got '{prediction_mode}'"
        )

    resolved_mode = prediction_mode
    if resolved_mode == "auto":
        resolved_mode = "ensemble" if run.checkpoint_count > 1 else "single"

    if resolved_mode == "ensemble":
        ensemble_path = _resolve_prediction_path(run, split, "ensemble")
        if ensemble_path is None:
            raise FileNotFoundError(
                f"Could not find {split} ensemble predictions for {run.analysis_dir}"
            )
        return pd.read_csv(ensemble_path)

    seed_path = _resolve_prediction_path(run, split, "seed")
    if seed_path is None:
        if split == "val" and run.checkpoint_count == 1:
            ensemble_path = _resolve_prediction_path(run, split, "ensemble")
            if ensemble_path is not None:
                return pd.read_csv(ensemble_path)
        raise FileNotFoundError(
            f"Could not find {split} seed predictions for {run.analysis_dir}"
        )

    seed_predictions = pd.read_csv(seed_path)
    if run.checkpoint_count <= 1:
        return seed_predictions

    checkpoint_paths = [
        str(resolve_repo_path(run.repo_root, path))
        for path in run.summary.get("checkpoints", [])
    ]
    if single_checkpoint_index < 0 or single_checkpoint_index >= len(checkpoint_paths):
        raise IndexError(
            f"single_checkpoint_index={single_checkpoint_index} is out of range for {len(checkpoint_paths)} checkpoints"
        )

    selected_path = checkpoint_paths[single_checkpoint_index]
    filtered = seed_predictions[seed_predictions["checkpoint_path"] == selected_path].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for checkpoint '{selected_path}' in {seed_path}")
    return filtered.reset_index(drop=True)


def collapse_to_accession_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    group_columns = {
        "gleason_label": ("gleason_label", "first"),
        "pirads_label": ("pirads_label", "first"),
        "grade_group": ("grade_group", "first"),
        "gleason_prob": ("gleason_prob", "mean"),
        "pirads_prob": ("pirads_prob", "mean"),
    }
    for column in [
        "patient_id",
        "study_datetime",
        "has_pretest_history",
        "num_pretest_accessions",
        "patient_total_accession_count",
        "patient_test_accession_count",
        "test_sequence_index",
        "is_first_test_accession",
        "is_last_test_accession",
    ]:
        if column in frame.columns:
            group_columns[column] = (column, "first")

    accession_level = (
        frame.groupby("accession_id", as_index=False)
        .agg(**group_columns)
        .sort_values("accession_id")
        .reset_index(drop=True)
    )
    accession_level["num_averaging_tags_used"] = (
        frame.groupby("accession_id")["averaging_tag"].nunique().reindex(accession_level["accession_id"]).to_numpy()
    )
    return accession_level


def collapse_to_accession_tag_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    group_columns = {
        "gleason_label": ("gleason_label", "first"),
        "pirads_label": ("pirads_label", "first"),
        "grade_group": ("grade_group", "first"),
        "gleason_prob": ("gleason_prob", "mean"),
        "pirads_prob": ("pirads_prob", "mean"),
    }
    for column in [
        "patient_id",
        "study_datetime",
        "has_pretest_history",
        "num_pretest_accessions",
        "patient_total_accession_count",
        "patient_test_accession_count",
        "test_sequence_index",
        "is_first_test_accession",
        "is_last_test_accession",
    ]:
        if column in frame.columns:
            group_columns[column] = (column, "first")

    accession_tag_level = (
        frame.groupby(["accession_id", "averaging_tag"], as_index=False)
        .agg(**group_columns)
        .sort_values(["accession_id", "averaging_tag"])
        .reset_index(drop=True)
    )
    accession_tag_level["num_averaging_tags_used"] = (
        frame.groupby("accession_id")["averaging_tag"].nunique().reindex(accession_tag_level["accession_id"]).to_numpy()
    )
    return accession_tag_level


def _safe_roc_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def _safe_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(brier_score_loss(labels, probs))


def _safe_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(log_loss(labels, clipped, labels=[0, 1]))


def compute_prediction_metrics(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    accession_level = collapse_to_accession_level(frame)
    overall = accession_level.dropna(subset=["gleason_label", "gleason_prob"]).copy()
    if not overall.empty:
        labels = overall["gleason_label"].astype(int).to_numpy()
        probs = overall["gleason_prob"].astype(float).to_numpy()
        rows.append(
            {
                "split": split,
                "scope": "overall",
                "averaging_tag": "Overall",
                "row_count": int(len(overall)),
                "accession_count": int(overall["accession_id"].nunique()),
                "auroc": _safe_roc_auc(labels, probs),
                "brier": _safe_brier(labels, probs),
                "log_loss": _safe_log_loss(labels, probs),
            }
        )

    tag_order = list(dict.fromkeys(frame["averaging_tag"].dropna().tolist()))
    for averaging_tag in tag_order:
        tag_rows = frame[
            (frame["averaging_tag"] == averaging_tag)
            & frame["gleason_label"].notna()
            & frame["gleason_prob"].notna()
        ].copy()
        if tag_rows.empty:
            continue
        labels = tag_rows["gleason_label"].astype(int).to_numpy()
        probs = tag_rows["gleason_prob"].astype(float).to_numpy()
        rows.append(
            {
                "split": split,
                "scope": "per_tag",
                "averaging_tag": averaging_tag,
                "row_count": int(len(tag_rows)),
                "accession_count": int(tag_rows["accession_id"].nunique()),
                "auroc": _safe_roc_auc(labels, probs),
                "brier": _safe_brier(labels, probs),
                "log_loss": _safe_log_loss(labels, probs),
            }
        )

    return pd.DataFrame(rows)


def validate_comparable_runs(
    left_run: AnalysisRun,
    right_run: AnalysisRun,
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
) -> None:
    left_split_details = left_run.summary.get("split_details", {})
    right_split_details = right_run.summary.get("split_details", {})
    if left_split_details != right_split_details:
        raise ValueError(
            "Split details differ between the two analysis runs; compare only like-for-like splits."
        )

    left_keys = set(
        zip(
            left_predictions["accession_id"].astype(str),
            left_predictions["averaging_tag"].fillna("none").astype(str),
        )
    )
    right_keys = set(
        zip(
            right_predictions["accession_id"].astype(str),
            right_predictions["averaging_tag"].fillna("none").astype(str),
        )
    )
    if left_keys != right_keys:
        raise ValueError(
            "Prediction row universe differs between the two runs; compare only identical accession/tag sets."
        )


def _prediction_confidence(probs: pd.Series, predicted_labels: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(predicted_labels.astype(int) == 1, probs.astype(float), 1.0 - probs.astype(float)),
        index=probs.index,
    )


def _build_prediction_comparison_table(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    *,
    id_columns: list[str],
    left_label: str,
    right_label: str,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    shared_columns = [
        *id_columns,
        "gleason_label",
        "pirads_label",
        "grade_group",
        "patient_id",
        "study_datetime",
        "has_pretest_history",
        "num_pretest_accessions",
        "patient_total_accession_count",
        "patient_test_accession_count",
        "test_sequence_index",
        "is_first_test_accession",
        "is_last_test_accession",
        "num_averaging_tags_used",
    ]
    left_optional = [column for column in shared_columns if column in left_frame.columns]
    right_optional = [column for column in shared_columns if column in right_frame.columns]

    comparison = left_frame[left_optional + [f"{left_label}_prob"]].merge(
        right_frame[right_optional + [f"{right_label}_prob"]],
        on=id_columns,
        how="inner",
        suffixes=("", "_right"),
        validate="one_to_one",
    )

    for column in shared_columns:
        if column in id_columns:
            continue
        right_column = f"{column}_right"
        if right_column in comparison.columns:
            left_values = comparison[column] if column in comparison.columns else None
            right_values = comparison[right_column]
            if left_values is None:
                comparison[column] = right_values
            else:
                left_missing = left_values.isna()
                if left_missing.any():
                    comparison.loc[left_missing, column] = right_values[left_missing]
                mismatched = (~left_missing) & right_values.notna() & (left_values != right_values)
                if mismatched.any():
                    raise ValueError(f"Mismatched '{column}' values while joining prediction rows.")
            comparison = comparison.drop(columns=[right_column])

    comparison = comparison.dropna(subset=["gleason_label", f"{left_label}_prob", f"{right_label}_prob"]).copy()
    comparison["gleason_label"] = comparison["gleason_label"].astype(int)
    comparison[f"{left_label}_pred"] = (comparison[f"{left_label}_prob"] >= decision_threshold).astype(int)
    comparison[f"{right_label}_pred"] = (comparison[f"{right_label}_prob"] >= decision_threshold).astype(int)
    comparison[f"{left_label}_correct"] = (
        comparison[f"{left_label}_pred"] == comparison["gleason_label"]
    )
    comparison[f"{right_label}_correct"] = (
        comparison[f"{right_label}_pred"] == comparison["gleason_label"]
    )
    comparison[f"{left_label}_confidence"] = _prediction_confidence(
        comparison[f"{left_label}_prob"], comparison[f"{left_label}_pred"]
    )
    comparison[f"{right_label}_confidence"] = _prediction_confidence(
        comparison[f"{right_label}_prob"], comparison[f"{right_label}_pred"]
    )
    comparison["prob_delta_right_minus_left"] = (
        comparison[f"{right_label}_prob"] - comparison[f"{left_label}_prob"]
    )
    comparison["abs_prob_delta"] = comparison["prob_delta_right_minus_left"].abs()

    comparison["comparison_category"] = "both_wrong"
    both_correct = comparison[f"{left_label}_correct"] & comparison[f"{right_label}_correct"]
    left_better = comparison[f"{left_label}_correct"] & ~comparison[f"{right_label}_correct"]
    right_better = ~comparison[f"{left_label}_correct"] & comparison[f"{right_label}_correct"]
    comparison.loc[both_correct, "comparison_category"] = "both_correct"
    comparison.loc[left_better, "comparison_category"] = f"{left_label}_correct_{right_label}_wrong"
    comparison.loc[right_better, "comparison_category"] = f"{left_label}_wrong_{right_label}_correct"

    sort_columns = ["comparison_category", "abs_prob_delta", *id_columns]
    return comparison.sort_values(sort_columns, ascending=[True, False, *([True] * len(id_columns))]).reset_index(
        drop=True
    )


def build_accession_comparison_table(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    left_accession = collapse_to_accession_level(left_predictions).rename(
        columns={
            "gleason_prob": f"{left_label}_prob",
            "pirads_prob": f"{left_label}_pirads_prob",
        }
    )
    right_accession = collapse_to_accession_level(right_predictions).rename(
        columns={
            "gleason_prob": f"{right_label}_prob",
            "pirads_prob": f"{right_label}_pirads_prob",
        }
    )
    return _build_prediction_comparison_table(
        left_accession,
        right_accession,
        id_columns=["accession_id"],
        left_label=left_label,
        right_label=right_label,
        decision_threshold=decision_threshold,
    )


def build_accession_tag_comparison_table(
    left_predictions: pd.DataFrame,
    right_predictions: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    left_accession_tag = collapse_to_accession_tag_level(left_predictions).rename(
        columns={
            "gleason_prob": f"{left_label}_prob",
            "pirads_prob": f"{left_label}_pirads_prob",
        }
    )
    right_accession_tag = collapse_to_accession_tag_level(right_predictions).rename(
        columns={
            "gleason_prob": f"{right_label}_prob",
            "pirads_prob": f"{right_label}_pirads_prob",
        }
    )
    return _build_prediction_comparison_table(
        left_accession_tag,
        right_accession_tag,
        id_columns=["accession_id", "averaging_tag"],
        left_label=left_label,
        right_label=right_label,
        decision_threshold=decision_threshold,
    )


def summarize_pairwise_shift(
    comparison_table: pd.DataFrame,
    *,
    left_label: str,
    right_label: str,
) -> pd.DataFrame:
    if comparison_table.empty:
        return pd.DataFrame()

    left_probs = comparison_table[f"{left_label}_prob"].astype(float)
    right_probs = comparison_table[f"{right_label}_prob"].astype(float)
    labels = comparison_table["gleason_label"].astype(int)
    delta = right_probs - left_probs

    rows = [
        {
            "metric": "pearson_prob_corr",
            "value": float(left_probs.corr(right_probs, method="pearson")),
        },
        {
            "metric": "spearman_prob_corr",
            "value": float(left_probs.corr(right_probs, method="spearman")),
        },
        {
            "metric": "mean_delta_all",
            "value": float(delta.mean()),
        },
        {
            "metric": "mean_delta_positive",
            "value": float(delta[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        },
        {
            "metric": "mean_delta_negative",
            "value": float(delta[labels == 0].mean()) if (labels == 0).any() else float("nan"),
        },
        {
            "metric": "mean_abs_delta",
            "value": float(delta.abs().mean()),
        },
    ]
    return pd.DataFrame(rows)


def select_disagreement_cases(
    comparison_table: pd.DataFrame,
    *,
    category: str,
    left_label: str,
    right_label: str,
    max_cases: int = 25,
) -> pd.DataFrame:
    if comparison_table.empty:
        return comparison_table.copy()

    subset = comparison_table[comparison_table["comparison_category"] == category].copy()
    if subset.empty:
        return subset

    if category == f"{left_label}_correct_{right_label}_wrong":
        subset["ranking_score"] = subset[f"{right_label}_confidence"]
    elif category == f"{left_label}_wrong_{right_label}_correct":
        subset["ranking_score"] = subset[f"{left_label}_confidence"]
    else:
        subset["ranking_score"] = np.maximum(
            subset[f"{left_label}_confidence"].astype(float),
            subset[f"{right_label}_confidence"].astype(float),
        )

    subset = subset.sort_values(
        ["ranking_score", "abs_prob_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return subset.head(max_cases)


@lru_cache(maxsize=16)
def _cached_split_lookup(config_path: str, experiment_name: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_args = load_base_args(Path(config_path), experiment_name)
    val_labels, test_labels, _, split_details, cohort_summary = build_context(base_args)
    return (
        {
            "base_args": base_args,
            "val": {label.accession_id: label for label in val_labels},
            "test": {label.accession_id: label for label in test_labels},
        },
        {
            "split_details": split_details,
            "cohort_summary": cohort_summary,
        },
    )


def _lookup_accession_label(run: AnalysisRun, accession_id: str, split: str):
    lookup_payload, _ = _cached_split_lookup(str(run.config_path), run.experiment_name)
    split_lookup = lookup_payload[split]
    if accession_id not in split_lookup:
        raise KeyError(f"Accession '{accession_id}' not found in {split} split for {run.analysis_dir}")
    return lookup_payload["base_args"], split_lookup[accession_id]


def _sample_to_numpy(sample: dict[str, Any], key: str) -> np.ndarray:
    tensor = sample[key].detach().cpu().numpy()
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Expected tensor shape (1, D, H, W) for '{key}', got {tensor.shape}")
    return tensor[0]


def load_case_review_bundle(
    left_run: AnalysisRun,
    right_run: AnalysisRun,
    *,
    accession_id: str,
    averaging_tag: str,
    split: str = "test",
) -> dict[str, Any]:
    left_source = str(left_run.run_metadata.get("image_source", left_run.base_args.get("image_source", "recon")))
    right_source = str(right_run.run_metadata.get("image_source", right_run.base_args.get("image_source", "recon")))

    left_base_args, left_label = _lookup_accession_label(left_run, accession_id, split)
    right_base_args, right_label = _lookup_accession_label(right_run, accession_id, split)

    def _load_sample(base_args: dict[str, Any], accession_label, *, coil_combine: str) -> dict[str, Any]:
        dataset = create_image_dataset(
            accession_labels=[accession_label],
            h5_root=Path(base_args["h5_root"]),
            image_source=str(base_args.get("image_source", "recon")),
            image_metrics=str(base_args["image_metrics"]),
            dicom_roots=[Path(path) for path in base_args.get("dicom_h5_roots", [])] or None,
            coil_combine=coil_combine,
            averaging_tag=averaging_tag,
            averaging_tags=[averaging_tag] if str(base_args.get("image_source", "recon")) in {"recon", "hybrid"} else None,
        )
        return dataset[0]

    if left_source == "recon" and right_source == "recon":
        left_sample = _load_sample(
            left_base_args,
            left_label,
            coil_combine=str(left_run.run_metadata.get("coil_combine", left_base_args.get("coil_combine", "rss"))),
        )
        right_sample = _load_sample(
            right_base_args,
            right_label,
            coil_combine=str(right_run.run_metadata.get("coil_combine", right_base_args.get("coil_combine", "rss"))),
        )
        return {
            "layout": "recon_vs_recon",
            "accession_id": accession_id,
            "averaging_tag": averaging_tag,
            "volumes": {
                "left_adc": _sample_to_numpy(left_sample, "adc"),
                "right_adc": _sample_to_numpy(right_sample, "adc"),
                "left_b1500": _sample_to_numpy(left_sample, "b1500"),
                "right_b1500": _sample_to_numpy(right_sample, "b1500"),
            },
        }

    hybrid_run = left_run if left_source == "hybrid" else right_run if right_source == "hybrid" else None
    hybrid_base_args = left_base_args if hybrid_run is left_run else right_base_args if hybrid_run is right_run else None
    hybrid_label = left_label if hybrid_run is left_run else right_label if hybrid_run is right_run else None
    if hybrid_run is None or hybrid_base_args is None or hybrid_label is None:
        raise ValueError(
            f"Unsupported comparison layout for image_source pair ({left_source}, {right_source})"
        )

    hybrid_sample = _load_sample(
        hybrid_base_args,
        hybrid_label,
        coil_combine=str(hybrid_run.run_metadata.get("coil_combine", hybrid_base_args.get("coil_combine", "rss"))),
    )
    return {
        "layout": "dwi_plus_t2",
        "accession_id": accession_id,
        "averaging_tag": averaging_tag,
        "volumes": {
            "axt2": _sample_to_numpy(hybrid_sample, "axt2"),
            "adc": _sample_to_numpy(hybrid_sample, "adc"),
            "b1500": _sample_to_numpy(hybrid_sample, "b1500"),
        },
    }


def default_review_tag(run: AnalysisRun) -> str:
    tags = run.summary.get("averaging_tags", [])
    if not tags:
        return "none"
    return str(tags[-1])
