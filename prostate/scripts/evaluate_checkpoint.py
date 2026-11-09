"""Evaluate saved checkpoints for both image and k-space models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prostate_kspace_classifier.data.h5_dataset import (
    DEFAULT_AVERAGING_TAG,
    DEFAULT_COIL_COMBINE,
    IMAGE_METRICS_CHOICES,
    IMAGE_SOURCE_CHOICES,
    ProstateH5Dataset,
    create_image_dataset,
    get_aligned_cohort,
    resolve_image_input_keys,
    validate_image_source_metrics,
)
from prostate_kspace_classifier.data.recipes import (
    DEFAULT_RECIPE_SEED,
    RECIPE_MODE_CHOICES,
)
from prostate_kspace_classifier.data.metadata_index import (
    consecutive_patient_holdout_split,
    list_dicom_accessions,
    load_accession_labels,
    load_accession_labels_from_csv,
    temporal_split,
)
from prostate_kspace_classifier.models.image_metrics import (
    MODEL_VARIANT_CHOICES,
    MODEL_VARIANT_SINGLE_STREAM,
    build_image_model,
)
from prostate_kspace_classifier.models.kspace_volume_classifier import (
    KSpaceVolumeClassifier,
)
from prostate_kspace_classifier.training.engine import evaluate
from prostate_kspace_classifier.training.seed import seed_everything
from prostate_kspace_classifier.utils.logging import setup_logging


def _get_config_value(config: dict, key: str, default):
    return config[key] if key in config else default


def _resolve_dicom_roots(cli_roots, config_roots):
    if cli_roots:
        return [Path(path) for group in cli_roots for path in group]
    if config_roots:
        return [Path(path) for path in config_roots]
    return None


def _uses_averaging_tags(image_source: str) -> bool:
    return image_source in {"recon", "hybrid"}


def _resolve_averaging_tags(
    cli_args: argparse.Namespace,
    config: dict,
    *,
    image_source: str,
) -> list[str]:
    if not _uses_averaging_tags(image_source):
        return []
    config_tags = config.get("averaging_tags")
    if config_tags:
        return list(config_tags)
    cli_tags = getattr(cli_args, "averaging_tags", None)
    if cli_tags:
        return list(cli_tags)
    return [_get_config_value(config, "averaging_tag", cli_args.averaging_tag)]


def _split_accession_ids(
    labels,
    *,
    split_strategy: str,
    split_test_frac: float,
    split_test_start_date: str | None,
    split_pool_train_frac: float,
):
    if split_strategy == "temporal":
        return temporal_split(labels, train_frac=0.75, val_frac=0.15)
    if split_strategy == "consecutive_patient_holdout":
        return consecutive_patient_holdout_split(
            labels,
            train_frac=split_pool_train_frac,
            test_frac=split_test_frac,
            test_start_date=split_test_start_date,
        )
    raise ValueError(f"Unsupported split strategy '{split_strategy}'")


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict, dict]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload.get("config", {})
    return payload, {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier checkpoint")
    parser.add_argument("--mode", type=str, choices=["image", "kspace"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument(
        "--pathology-csv",
        type=Path,
        default=None,
        help="Optional pathology CSV containing highest_grade_group per mri_accession.",
    )
    parser.add_argument(
        "--gleason-positive-grade-group-threshold",
        type=int,
        default=2,
        help="Define positive Gleason target as grade_group >= threshold (default: 2).",
    )
    parser.add_argument(
        "--image-source", type=str, choices=IMAGE_SOURCE_CHOICES, default="recon"
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=MODEL_VARIANT_CHOICES,
        default=MODEL_VARIANT_SINGLE_STREAM,
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=["recon_only", "dicom_only", "aligned"],
        default=None,
    )
    parser.add_argument(
        "--image-metrics",
        type=str,
        choices=IMAGE_METRICS_CHOICES,
        default="adc_b1500",
        help="Image metric selection for the requested source (e.g. recon adc_b1500 or DICOM axt2).",
    )
    parser.add_argument(
        "--dicom-h5-roots",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="One or more directories containing DICOM-derived ADC/b1500 H5 files",
    )
    parser.add_argument(
        "--coil-combine",
        type=str,
        choices=["esc", "rss", "espirit"],
        default=DEFAULT_COIL_COMBINE,
    )
    parser.add_argument("--averaging-tag", type=str, default=DEFAULT_AVERAGING_TAG)
    parser.add_argument(
        "--averaging-tags",
        type=str,
        action="append",
        default=None,
        help="Repeat flag to evaluate a recon checkpoint across multiple averaging tags.",
    )
    parser.add_argument(
        "--recipe-mode",
        type=str,
        choices=RECIPE_MODE_CHOICES,
        default=None,
    )
    parser.add_argument("--recipes-per-accession", type=int, default=None)
    parser.add_argument(
        "--recipe-family-quotas",
        type=str,
        action="append",
        default=None,
    )
    parser.add_argument("--recipe-seed", type=int, default=DEFAULT_RECIPE_SEED)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--filter-mode", type=str, default="single_channel")
    parser.add_argument("--num-kspace-filters", type=int, default=8)
    parser.add_argument("--date-buffer-days", type=int, default=7)
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["temporal", "consecutive_patient_holdout"],
        default="temporal",
        help="Split strategy used during training.",
    )
    parser.add_argument(
        "--split-test-frac",
        type=float,
        default=0.10,
        help="For consecutive_patient_holdout: most-recent accession fraction used to anchor test start.",
    )
    parser.add_argument(
        "--split-test-start-date",
        type=str,
        default=None,
        help="For consecutive_patient_holdout: explicit test-start date (e.g., 2024-01-01). Overrides split-test-frac.",
    )
    parser.add_argument(
        "--split-pool-train-frac",
        type=float,
        default=0.85,
        help="For consecutive_patient_holdout: fraction of pre-test pool patients assigned to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    seed_everything(args.seed)
    desired_device = torch.device(args.device)
    device = (
        desired_device
        if (desired_device.type == "cpu" or torch.cuda.is_available())
        else torch.device("cpu")
    )

    state_dict, ckpt_config = _load_checkpoint(args.checkpoint, device)
    if ckpt_config:
        ckpt_mode = ckpt_config.get("mode")
        if ckpt_mode and ckpt_mode != args.mode:
            raise ValueError(
                f"Checkpoint mode '{ckpt_mode}' does not match --mode '{args.mode}'"
            )

    model_cfg = ckpt_config.get("model", {})
    data_cfg = ckpt_config.get("data", {})
    training_cfg = ckpt_config.get("training", {})
    cohort = _get_config_value(data_cfg, "cohort", args.cohort)
    if cohort is None:
        cohort = "recon_only"
    split_strategy = _get_config_value(data_cfg, "split_strategy", args.split_strategy)
    split_test_frac = float(
        _get_config_value(data_cfg, "split_test_frac", args.split_test_frac)
    )
    split_test_start_date = _get_config_value(
        data_cfg, "split_test_start_date", args.split_test_start_date
    )
    split_pool_train_frac = float(
        _get_config_value(data_cfg, "split_pool_train_frac", args.split_pool_train_frac)
    )

    pathology_csv_value = _get_config_value(
        data_cfg,
        "pathology_csv",
        str(args.pathology_csv) if args.pathology_csv else None,
    )
    pathology_csv = Path(pathology_csv_value) if pathology_csv_value else None
    gleason_positive_threshold = _get_config_value(
        data_cfg,
        "gleason_positive_grade_group_threshold",
        args.gleason_positive_grade_group_threshold,
    )
    enable_gleason_task = bool(
        _get_config_value(
            model_cfg,
            "enable_gleason_head",
            _get_config_value(training_cfg, "enable_gleason_task", False),
        )
    )
    pirads_loss_weight = float(
        _get_config_value(training_cfg, "pirads_loss_weight", 1.0)
    )
    gleason_loss_weight = float(
        _get_config_value(training_cfg, "gleason_loss_weight", 1.0)
    )

    if args.mode == "image":
        image_source = _get_config_value(data_cfg, "image_source", args.image_source)
        if image_source is None:
            image_source = "recon"
        model_variant = _get_config_value(
            model_cfg, "model_variant", args.model_variant
        )
        image_metrics = _get_config_value(data_cfg, "image_metrics", args.image_metrics)
        validate_image_source_metrics(image_source, image_metrics)
        image_keys = list(resolve_image_input_keys(image_source, image_metrics))
        dicom_roots = _resolve_dicom_roots(
            args.dicom_h5_roots, data_cfg.get("dicom_h5_roots")
        )

        if cohort == "dicom_only" and image_source != "dicom":
            raise ValueError("--cohort=dicom_only requires --image-source=dicom")

        # Match training split policy:
        # - aligned: evaluate on aligned test split
        # - recon_only/dicom_only: evaluate on the same aligned holdout test split
        if cohort in {"aligned", "recon_only", "dicom_only"} and dicom_roots is None:
            raise ValueError(
                "--dicom-h5-roots required for aligned-holdout evaluation policy"
            )

        if cohort == "dicom_only":
            source_labels = load_accession_labels_from_csv(
                args.labels_csv,
                pathology_csv=pathology_csv,
                gleason_positive_threshold=gleason_positive_threshold,
            )
            dicom_ids = list_dicom_accessions(dicom_roots)
            source_labels = [
                lbl for lbl in source_labels if lbl.accession_id in dicom_ids
            ]
            # Build aligned reference split from recon-matched labels, as done in training.
            alignment_base_labels = load_accession_labels(
                csv_path=args.labels_csv,
                h5_root=args.h5_root,
                date_buffer_days=args.date_buffer_days,
                pathology_csv=pathology_csv,
                gleason_positive_threshold=gleason_positive_threshold,
            )
            if not source_labels:
                raise ValueError("No labeled DICOM accessions found after filtering.")
        else:
            source_labels = load_accession_labels(
                csv_path=args.labels_csv,
                h5_root=args.h5_root,
                date_buffer_days=args.date_buffer_days,
                pathology_csv=pathology_csv,
                gleason_positive_threshold=gleason_positive_threshold,
            )
            alignment_base_labels = source_labels

        if cohort == "aligned":
            aligned_labels, _ = get_aligned_cohort(
                accession_labels=alignment_base_labels,
                recon_h5_root=args.h5_root,
                dicom_h5_roots=dicom_roots,
                require_raw_kspace=False,
            )
            _, _, test_ids = _split_accession_ids(
                aligned_labels,
                split_strategy=split_strategy,
                split_test_frac=split_test_frac,
                split_test_start_date=split_test_start_date,
                split_pool_train_frac=split_pool_train_frac,
            )
            id_map = {lbl.accession_id: lbl for lbl in aligned_labels}
            test_labels = [id_map[i] for i in test_ids]
        elif cohort in {"recon_only", "dicom_only"}:
            aligned_labels, _ = get_aligned_cohort(
                accession_labels=alignment_base_labels,
                recon_h5_root=args.h5_root,
                dicom_h5_roots=dicom_roots,
                require_raw_kspace=False,
            )
            _, _, aligned_test_ids = _split_accession_ids(
                aligned_labels,
                split_strategy=split_strategy,
                split_test_frac=split_test_frac,
                split_test_start_date=split_test_start_date,
                split_pool_train_frac=split_pool_train_frac,
            )
            aligned_id_map = {lbl.accession_id: lbl for lbl in aligned_labels}
            test_labels = [aligned_id_map[i] for i in aligned_test_ids]
        else:
            raise ValueError(f"Unsupported cohort '{cohort}' for image mode")

        dual_encoder = bool(_get_config_value(model_cfg, "dual_encoder", False))
        backbone = _get_config_value(model_cfg, "backbone", "r3d_18")
        backbone_pretrained = _get_config_value(model_cfg, "backbone_pretrained", False)
        dropout = _get_config_value(model_cfg, "dropout", 0.3)
        fusion_gate = bool(_get_config_value(model_cfg, "fusion_gate", False))
        coil_combine = _get_config_value(data_cfg, "coil_combine", args.coil_combine)
        averaging_tags = _resolve_averaging_tags(
            args, data_cfg, image_source=image_source
        )
        recipe_mode = _get_config_value(data_cfg, "recipe_mode", args.recipe_mode)
        recipe_family_quotas = _get_config_value(
            data_cfg,
            "recipe_family_quotas",
            args.recipe_family_quotas,
        )
        recipes_per_accession = _get_config_value(
            data_cfg,
            "recipes_per_accession",
            args.recipes_per_accession,
        )
        recipe_seed = int(
            _get_config_value(data_cfg, "recipe_seed", args.recipe_seed)
        )

        dataset = create_image_dataset(
            accession_labels=test_labels,
            h5_root=args.h5_root,
            image_source=image_source,
            image_metrics=image_metrics,
            dicom_roots=dicom_roots,
            coil_combine=coil_combine,
            averaging_tag=averaging_tags[0] if averaging_tags else args.averaging_tag,
            averaging_tags=averaging_tags,
            recipe_mode=recipe_mode,
            recipes_per_accession=recipes_per_accession,
            recipe_family_quotas=recipe_family_quotas,
            recipe_seed=recipe_seed,
            recipe_split="test",
        )
        model = build_image_model(
            model_variant=model_variant,
            input_keys=image_keys,
            dropout=dropout,
            backbone=backbone,
            backbone_pretrained=backbone_pretrained,
            dual_encoder=dual_encoder,
            freeze_backbone=bool(
                _get_config_value(model_cfg, "freeze_backbone", False)
            ),
            enable_gleason_head=enable_gleason_task,
            fusion_gate=fusion_gate,
        )
    else:
        dicom_roots = _resolve_dicom_roots(
            args.dicom_h5_roots, data_cfg.get("dicom_h5_roots")
        )
        if cohort == "dicom_only":
            raise ValueError(
                "--cohort=dicom_only is not supported for kspace evaluation"
            )

        labels = load_accession_labels(
            csv_path=args.labels_csv,
            h5_root=args.h5_root,
            date_buffer_days=args.date_buffer_days,
            pathology_csv=pathology_csv,
            gleason_positive_threshold=gleason_positive_threshold,
        )
        if cohort == "aligned":
            if dicom_roots is None:
                raise ValueError(
                    "--dicom-h5-roots required when evaluating aligned cohort"
                )
            labels, _ = get_aligned_cohort(
                accession_labels=labels,
                recon_h5_root=args.h5_root,
                dicom_h5_roots=dicom_roots,
                require_raw_kspace=True,
            )
        elif cohort != "recon_only":
            raise ValueError(f"Unsupported cohort '{cohort}' for kspace mode")

        _, _, test_ids = _split_accession_ids(
            labels,
            split_strategy=split_strategy,
            split_test_frac=split_test_frac,
            split_test_start_date=split_test_start_date,
            split_pool_train_frac=split_pool_train_frac,
        )
        id_map = {lbl.accession_id: lbl for lbl in labels}
        test_labels = [id_map[i] for i in test_ids]

        filter_mode = _get_config_value(model_cfg, "filter_mode", args.filter_mode)
        num_kspace_filters = _get_config_value(
            model_cfg, "num_kspace_filters", args.num_kspace_filters
        )
        backbone = _get_config_value(model_cfg, "backbone", "r3d_18")
        backbone_pretrained = _get_config_value(model_cfg, "backbone_pretrained", False)

        dataset = ProstateH5Dataset(
            accession_labels=test_labels,
            h5_root=args.h5_root,
            mode="kspace_post_grappa",
        )
        sample = dataset[0]
        num_channels = sample["kspace"].shape[0]
        model = KSpaceVolumeClassifier(
            num_input_channels=num_channels,
            num_kspace_filters=num_kspace_filters,
            backbone=backbone,
            backbone_pretrained=backbone_pretrained,
            filter_mode=filter_mode,
            enable_gleason_head=enable_gleason_task,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model.to(device)
    model.load_state_dict(state_dict)

    metrics = evaluate(
        model,
        dataloader,
        device,
        pirads_loss_weight=pirads_loss_weight,
        gleason_loss_weight=gleason_loss_weight if enable_gleason_task else 0.0,
    )
    print(metrics)


if __name__ == "__main__":
    main()
