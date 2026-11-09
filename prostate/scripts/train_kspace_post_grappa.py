"""Train the k-space classifier with FFTConv slice layers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader

from prostate_kspace_classifier.data.h5_dataset import ProstateH5Dataset, get_aligned_cohort
from prostate_kspace_classifier.data.metadata_index import (
    load_accession_labels,
    summarize_label_distribution,
    temporal_split,
)
from prostate_kspace_classifier.models.kspace_volume_classifier import KSpaceVolumeClassifier
from prostate_kspace_classifier.training.engine import evaluate, train_one_epoch
from prostate_kspace_classifier.training.schedulers import create_epoch_scheduler
from prostate_kspace_classifier.training.seed import seed_everything
from prostate_kspace_classifier.training.wandb_logging import WandbLogger
from prostate_kspace_classifier.utils.logging import setup_logging


DEFAULT_WANDB_PROJECT = "prostate-kspace-classifier"
COHORT_CHOICES = ("recon_only", "aligned")


def default_run_name(args: argparse.Namespace, *, num_channels: int) -> str:
    suffix = [
        "kspace",
        args.cohort,
        "mtl" if args.enable_gleason_task else "pirads",
        args.filter_mode,
        f"nf{args.num_kspace_filters}",
        f"ch{num_channels}",
        args.backbone,
        "pt" if args.backbone_pretrained else "scratch",
        f"lr{args.lr:.0e}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
    ]
    return "-".join(suffix)


def _flatten_split_stats(prefix: str, summary: Dict[str, object]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{prefix}/{sub_key}"] = float(sub_value)
        else:
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def _print_split_stats(split_name: str, summary: Dict[str, object]) -> None:
    print(
        f"[labels:{split_name}] n={summary['count']} "
        f"pirads_pos={summary['pirads_positive']} pirads_neg={summary['pirads_negative']} "
        f"gleason_labeled={summary['gleason_labeled_count']} "
        f"gleason_pos={summary['gleason_positive']} gleason_neg={summary['gleason_negative']} "
        f"pirads_dist={summary['pirads_distribution']} grade_group_dist={summary['grade_group_distribution']}"
    )


def build_checkpoint_config(
    args: argparse.Namespace,
    *,
    cohort_mode: str,
    dicom_roots: Sequence[Path] | None,
) -> dict:
    dicom_paths = [str(path) for path in dicom_roots] if dicom_roots else None
    return {
        "version": 1,
        "mode": "kspace",
        "model": {
            "filter_mode": args.filter_mode,
            "num_kspace_filters": args.num_kspace_filters,
            "backbone": args.backbone,
            "backbone_pretrained": args.backbone_pretrained,
            "enable_gleason_head": args.enable_gleason_task,
        },
        "data": {
            "cohort": cohort_mode,
            "dicom_h5_roots": dicom_paths,
            "pathology_csv": str(args.pathology_csv) if args.pathology_csv else None,
            "gleason_positive_grade_group_threshold": args.gleason_positive_grade_group_threshold,
        },
        "training": {
            "enable_gleason_task": args.enable_gleason_task,
            "primary_metric": args.primary_metric,
            "pirads_loss_weight": args.pirads_loss_weight,
            "gleason_loss_weight": args.gleason_loss_weight,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train k-space classifier")
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument(
        "--dicom-h5-roots",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="Optional DICOM H5 roots; required for --cohort=aligned",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        choices=COHORT_CHOICES,
        default="recon_only",
        help="Cohort selection: recon_only or aligned (dicom_only is not supported for k-space).",
    )
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument(
        "--pathology-csv",
        type=Path,
        default=None,
        help="Optional pathology CSV containing highest_grade_group per mri_accession.",
    )
    parser.add_argument(
        "--enable-gleason-task",
        action="store_true",
        help="Enable multitask training with auxiliary Gleason/grade-group supervision.",
    )
    parser.add_argument(
        "--gleason-positive-grade-group-threshold",
        type=int,
        default=2,
        help="Define positive Gleason target as grade_group >= threshold (default: 2).",
    )
    parser.add_argument("--filter-mode", type=str, choices=["single_channel", "multi_channel"], default="single_channel")
    parser.add_argument("--num-kspace-filters", type=int, default=8)
    parser.add_argument("--backbone", type=str, choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--backbone-pretrained", action="store_true", help="Use Kinetics pretrained weights for the backbone")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--pirads-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for PI-RADS/csPCa head.",
    )
    parser.add_argument(
        "--gleason-loss-weight",
        type=float,
        default=1.0,
        help="Loss weight for Gleason auxiliary head when enabled.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        choices=["auroc", "pirads_auroc", "gleason_auroc"],
        default="auroc",
        help="Metric used for checkpoint selection and early stopping.",
    )
    parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine"], default="cosine")
    parser.add_argument("--lr-warmup-epochs", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval-steps", type=int, default=0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
        help="Stop if validation primary metric does not improve for this many epochs (0 disables early stopping)",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of minibatches to accumulate before an optimizer step")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--date-buffer-days", type=int, default=7)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None, help="Logical experiment group name in W&B")
    parser.add_argument("--wandb-job-type", type=str, default=None, help="Sub-experiment stage label in W&B")
    parser.add_argument(
        "--wandb-tags",
        type=str,
        action="append",
        default=None,
        help="Repeat flag to attach tags in W&B (e.g. --wandb-tags kspace --wandb-tags seed_22222)",
    )
    parser.add_argument(
        "--wandb-experiment-id",
        type=str,
        default=None,
        help="Stable experiment identifier stored in W&B config for API filtering",
    )
    parser.add_argument("--wandb-mode", type=str, default="online")
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> None:
    setup_logging()
    seed_everything(args.seed)
    desired_device = torch.device(args.device)
    device = desired_device if (desired_device.type == "cpu" or torch.cuda.is_available()) else torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = args.primary_metric
    if args.enable_gleason_task and primary_metric == "auroc":
        primary_metric = "gleason_auroc"

    cohort_mode = args.cohort

    labels = load_accession_labels(
        csv_path=args.labels_csv,
        h5_root=args.h5_root,
        date_buffer_days=args.date_buffer_days,
        pathology_csv=args.pathology_csv,
        gleason_positive_threshold=args.gleason_positive_grade_group_threshold,
    )
    dicom_roots = [Path(p) for group in args.dicom_h5_roots for p in group] if args.dicom_h5_roots else None
    checkpoint_config = build_checkpoint_config(
        args,
        cohort_mode=cohort_mode,
        dicom_roots=dicom_roots,
    )
    checkpoint_config["training"]["primary_metric"] = primary_metric

    # Align cohort if requested
    cohort_stats = None
    if cohort_mode == "aligned":
        if dicom_roots is None:
            raise ValueError("--dicom-h5-roots required when using --cohort=aligned")
        labels, cohort_stats = get_aligned_cohort(
            accession_labels=labels,
            recon_h5_root=args.h5_root,
            dicom_h5_roots=dicom_roots,
            require_raw_kspace=True,  # K-space baseline needs raw k-space stored
        )
        print(
            f"[aligned_cohort] labeled={cohort_stats.labeled_count} "
            f"raw_kspace={cohort_stats.raw_kspace_count} dicom={cohort_stats.dicom_count} "
            f"-> aligned={cohort_stats.aligned_count}"
        )

    train_ids, val_ids, _ = temporal_split(labels, train_frac=0.75, val_frac=0.15)
    print(
        f"[split] train={len(train_ids)} val={len(val_ids)} source=kspace cohort={cohort_mode}"
    )
    id_map = {lbl.accession_id: lbl for lbl in labels}
    train_labels = [id_map[i] for i in train_ids]
    val_labels = [id_map[i] for i in val_ids]
    train_summary = summarize_label_distribution(train_labels)
    val_summary = summarize_label_distribution(val_labels)
    _print_split_stats("train", train_summary)
    _print_split_stats("val", val_summary)

    train_dataset = ProstateH5Dataset(
        accession_labels=train_labels,
        h5_root=args.h5_root,
        mode="kspace_post_grappa",
    )
    val_dataset = ProstateH5Dataset(
        accession_labels=val_labels,
        h5_root=args.h5_root,
        mode="kspace_post_grappa",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample = train_dataset[0]
    num_channels = sample["kspace"].shape[0]
    model = KSpaceVolumeClassifier(
        num_input_channels=num_channels,
        num_kspace_filters=args.num_kspace_filters,
        backbone=args.backbone,
        backbone_pretrained=args.backbone_pretrained,
        filter_mode=args.filter_mode,
        enable_gleason_head=args.enable_gleason_task,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    lr_schedule = create_epoch_scheduler(
        args.lr_scheduler,
        total_epochs=args.epochs,
        warmup_epochs=args.lr_warmup_epochs,
    )

    def _set_epoch_lr(epoch_idx: int) -> float:
        if lr_schedule is None:
            return optimizer.param_groups[0]["lr"]
        multiplier = lr_schedule(epoch_idx)
        for base_lr, param_group in zip(base_lrs, optimizer.param_groups):
            param_group["lr"] = base_lr * multiplier
        return optimizer.param_groups[0]["lr"]

    run_name = args.wandb_run_name or default_run_name(args, num_channels=num_channels)
    checkpoint_stem = default_run_name(args, num_channels=num_channels)
    wandb_config = {
        "experiment": "kspace_post_grappa",
        "filter_mode": args.filter_mode,
        "num_kspace_filters": args.num_kspace_filters,
        "backbone": args.backbone,
        "backbone_pretrained": args.backbone_pretrained,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "eval_interval_steps": args.eval_interval_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "cohort": cohort_mode,
        "pathology_csv": str(args.pathology_csv) if args.pathology_csv else None,
        "enable_gleason_task": args.enable_gleason_task,
        "gleason_positive_grade_group_threshold": args.gleason_positive_grade_group_threshold,
        "pirads_loss_weight": args.pirads_loss_weight,
        "gleason_loss_weight": args.gleason_loss_weight,
        "primary_metric": primary_metric,
    }
    if args.wandb_experiment_id:
        wandb_config["experiment_id"] = args.wandb_experiment_id
    if args.wandb_group:
        wandb_config["wandb_group"] = args.wandb_group
    if args.wandb_job_type:
        wandb_config["wandb_job_type"] = args.wandb_job_type
    if args.wandb_tags:
        wandb_config["wandb_tags"] = list(args.wandb_tags)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        config=wandb_config,
        tags=args.wandb_tags,
    )
    wandb_logger.log(
        {
            "data/train_count": len(train_ids),
            "data/val_count": len(val_ids),
            "data/image_source": "kspace",
            "data/cohort": cohort_mode,
            "data/aligned_cohort": cohort_mode == "aligned",
            "data/enable_gleason_task": args.enable_gleason_task,
            "data/primary_metric": primary_metric,
        },
        step=0,
    )
    wandb_logger.log(_flatten_split_stats("splits/train", train_summary), step=0)
    wandb_logger.log(_flatten_split_stats("splits/val", val_summary), step=0)

    best_primary_metric = -1.0
    last_improvement_epoch = 0
    current_epoch = 0
    best_path = args.output_dir / f"best_{checkpoint_stem}.pt"
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)

    def _fmt(metric: float) -> str:
        return f"{metric:.4f}" if math.isfinite(metric) else "nan"

    def _log_val_metrics(step: int, val_metrics: Dict[str, float], *, context: str) -> None:
        nonlocal best_primary_metric, last_improvement_epoch
        current_primary = float(val_metrics.get(primary_metric, float("nan")))
        if math.isfinite(current_primary) and current_primary > best_primary_metric:
            best_primary_metric = current_primary
            last_improvement_epoch = current_epoch
            torch.save({"state_dict": model.state_dict(), "config": checkpoint_config}, best_path)
            print(
                f"Saved new best checkpoint to {best_path} "
                f"(context={context}, step={step}, {primary_metric}={_fmt(current_primary)})"
            )

        log_payload: Dict[str, Any] = {
            "val/context": context,
            "val/primary_metric_value": current_primary,
            "val/best_primary_metric": best_primary_metric,
        }
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                log_payload[f"val/{key}"] = float(value)
        wandb_logger.log(log_payload, step=step)

    def _mid_epoch_eval(global_step: int) -> None:
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight if args.enable_gleason_task else 0.0,
        )
        print(
            f"[val] step {global_step}: "
            f"pirads_auroc={_fmt(float(val_metrics.get('pirads_auroc', float('nan'))))} "
            f"gleason_auroc={_fmt(float(val_metrics.get('gleason_auroc', float('nan'))))} "
            f"best_{primary_metric}={_fmt(best_primary_metric)}"
        )
        _log_val_metrics(global_step, val_metrics, context="mid_epoch")

    for epoch in range(1, args.epochs + 1):
        current_epoch = epoch
        epoch_step = epoch * steps_per_epoch
        current_lr = _set_epoch_lr(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            log_interval=args.log_interval,
            progress_bar=True,
            logger=wandb_logger,
            epoch=epoch,
            base_step=(epoch - 1) * steps_per_epoch,
            eval_interval_steps=args.eval_interval_steps,
            eval_callback=_mid_epoch_eval if args.eval_interval_steps else None,
            gradient_accumulation_steps=args.grad_accum_steps,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight if args.enable_gleason_task else 0.0,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            pirads_loss_weight=args.pirads_loss_weight,
            gleason_loss_weight=args.gleason_loss_weight if args.enable_gleason_task else 0.0,
        )
        _log_val_metrics(epoch_step, val_metrics, context="epoch_end")
        print(
            f"Epoch {epoch}: "
            f"train_pirads_auroc={_fmt(float(train_metrics.get('pirads_auroc', float('nan'))))} "
            f"train_gleason_auroc={_fmt(float(train_metrics.get('gleason_auroc', float('nan'))))} "
            f"val_pirads_auroc={_fmt(float(val_metrics.get('pirads_auroc', float('nan'))))} "
            f"val_gleason_auroc={_fmt(float(val_metrics.get('gleason_auroc', float('nan'))))} "
            f"best_{primary_metric}={_fmt(best_primary_metric)}"
        )
        train_log_payload: Dict[str, Any] = {"epoch": epoch, "train/lr": current_lr}
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                train_log_payload[f"train/{key}"] = float(value)
        wandb_logger.log(train_log_payload, step=epoch_step)
        if args.early_stopping_patience > 0 and best_primary_metric >= 0:
            epochs_since_best = epoch - last_improvement_epoch
            if epochs_since_best >= args.early_stopping_patience:
                print(
                    f"[early-stop] no {primary_metric} improvement for {epochs_since_best} epochs "
                    f"(patience={args.early_stopping_patience}); stopping at epoch {epoch}"
                )
                break

    wandb_logger.set_summary(
        {
            "best_primary_metric_name": primary_metric,
            "best_primary_metric": best_primary_metric,
        }
    )
    wandb_logger.finish()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
