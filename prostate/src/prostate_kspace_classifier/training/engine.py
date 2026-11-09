"""Training utilities providing BCE + AUROC loops."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import compute_metrics
from .losses import masked_bce_with_logits


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _unpack_model_outputs(
    outputs: torch.Tensor | Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if torch.is_tensor(outputs):
        return outputs, None
    if isinstance(outputs, dict):
        return outputs["pirads"], outputs.get("gleason")
    raise TypeError(f"Unsupported model output type: {type(outputs)!r}")


def _merge_task_metrics(
    *,
    avg_total_loss: float,
    avg_pirads_loss: float,
    avg_gleason_loss: float,
    pirads_metrics: Dict[str, float],
    gleason_metrics: Dict[str, float],
    gleason_labeled_count: int,
    saw_gleason_head: bool,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "loss": avg_total_loss,
        "pirads_loss": avg_pirads_loss,
        "auroc": pirads_metrics["auroc"],
        "accuracy": pirads_metrics["accuracy"],
        "pirads_auroc": pirads_metrics["auroc"],
        "pirads_accuracy": pirads_metrics["accuracy"],
    }
    if not saw_gleason_head:
        return metrics

    metrics.update(
        {
            "gleason_loss": avg_gleason_loss,
            "gleason_auroc": gleason_metrics["auroc"],
            "gleason_accuracy": gleason_metrics["accuracy"],
            "gleason_labeled_count": float(gleason_labeled_count),
        }
    )
    return metrics


def _compute_task_metrics(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    accession_ids: list[str],
) -> tuple[Dict[str, float], int]:
    if logits.numel() == 0 or targets.numel() == 0:
        return compute_metrics(logits, targets), 0
    if not accession_ids:
        return compute_metrics(logits, targets), int(len(targets))
    if len(accession_ids) != len(targets):
        raise ValueError(
            f"Expected {len(targets)} accession IDs, got {len(accession_ids)}"
        )

    logits_cpu = logits.detach().to(dtype=torch.float32, device="cpu").view(-1)
    targets_cpu = targets.detach().to(dtype=torch.float32, device="cpu").view(-1)
    if len(set(accession_ids)) == len(accession_ids):
        return compute_metrics(logits_cpu, targets_cpu), int(len(targets_cpu))

    probs_cpu = torch.sigmoid(logits_cpu)
    grouped: dict[str, tuple[float, float, int]] = {}
    for accession_id, prob, target in zip(
        accession_ids,
        probs_cpu.tolist(),
        targets_cpu.tolist(),
    ):
        if accession_id in grouped:
            prob_sum, prev_target, count = grouped[accession_id]
            if not math.isclose(prev_target, target, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"Inconsistent labels found while aggregating accession '{accession_id}'"
                )
            grouped[accession_id] = (prob_sum + prob, prev_target, count + 1)
        else:
            grouped[accession_id] = (prob, target, 1)

    aggregated_probs = torch.tensor(
        [prob_sum / count for prob_sum, _, count in grouped.values()],
        dtype=torch.float32,
    )
    aggregated_targets = torch.tensor(
        [target for _, target, _ in grouped.values()],
        dtype=torch.float32,
    )
    metrics = compute_metrics(
        aggregated_probs,
        aggregated_targets,
        from_logits=False,
    )
    return metrics, int(len(aggregated_targets))


def _group_metrics_by_averaging_tag(
    *,
    task_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    averaging_tags: list[str],
    count_key_suffix: str,
) -> Dict[str, float]:
    if not averaging_tags:
        return {}
    if len(averaging_tags) != len(targets):
        raise ValueError(
            f"Expected {len(targets)} averaging tags for {task_name}, got {len(averaging_tags)}"
        )

    grouped_metrics: Dict[str, float] = {}
    for averaging_tag in dict.fromkeys(averaging_tags):
        mask = torch.tensor(
            [tag == averaging_tag for tag in averaging_tags],
            dtype=torch.bool,
        )
        tag_metrics = compute_metrics(logits[mask], targets[mask])
        prefix = f"average_tag/{averaging_tag}"
        grouped_metrics[f"{prefix}/{task_name}_{count_key_suffix}"] = float(
            mask.sum().item()
        )
        grouped_metrics[f"{prefix}/{task_name}_auroc"] = tag_metrics["auroc"]
        grouped_metrics[f"{prefix}/{task_name}_accuracy"] = tag_metrics["accuracy"]
    return grouped_metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    *,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    pirads_loss_weight: float = 1.0,
    gleason_loss_weight: float = 1.0,
    gleason_loss_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    log_interval: int = 50,
    progress_bar: bool = False,
    logger: Optional[Any] = None,
    epoch: Optional[int] = None,
    base_step: int = 0,
    eval_interval_steps: int = 0,
    eval_callback: Optional[Callable[[int], None]] = None,
    gradient_accumulation_steps: int = 1,
    amp_dtype: torch.dtype | None = None,
) -> Dict[str, float]:
    model.train()
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    total_loss = 0.0
    pirads_loss_total = 0.0
    gleason_loss_total = 0.0
    gleason_label_count = 0
    all_pirads_logits = []
    all_pirads_labels = []
    all_pirads_accession_ids: list[str] = []
    all_gleason_logits = []
    all_gleason_labels = []
    all_gleason_accession_ids: list[str] = []
    saw_gleason_head = False
    interval_loss = 0.0
    interval_samples = 0
    total_steps = len(dataloader)
    # Handle cases where the last accumulation group is smaller than gradient_accumulation_steps.
    final_accum_steps = (
        total_steps % gradient_accumulation_steps or gradient_accumulation_steps
    )
    total_optimizer_steps = (
        total_steps + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    pbar = None
    iterable = dataloader
    if progress_bar:
        desc = f"Train Epoch {epoch}" if epoch is not None else "Train"
        pbar = tqdm(dataloader, desc=desc, total=total_steps, leave=False)
        iterable = pbar

    def _log_step(step_idx: int, avg_loss: float, global_step: int) -> None:
        if progress_bar and pbar is not None:
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        else:
            print(
                f"[train] step {step_idx + 1:>4}/{total_optimizer_steps} | "
                f"loss={avg_loss:.4f}"
            )
        if logger is not None:
            logger.log({"train/step_loss": avg_loss}, step=global_step)

    optimizer.zero_grad()
    optimizer_steps = 0
    _pirads_loss_fn = (
        loss_fn if loss_fn is not None else F.binary_cross_entropy_with_logits
    )
    _gleason_loss_fn = (
        gleason_loss_fn if gleason_loss_fn is not None else masked_bce_with_logits
    )
    for step, batch in enumerate(iterable):
        micro_step = step + 1
        labels = batch["label"].to(device)
        batch_accession_ids = list(batch.get("accession_id", []))
        if batch_accession_ids and len(batch_accession_ids) != labels.size(0):
            raise ValueError(
                f"Expected {labels.size(0)} accession IDs in batch, got {len(batch_accession_ids)}"
            )
        batch_on_device = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

        with _autocast_context(device, amp_dtype):
            outputs = model(batch_on_device)
            pirads_logits, gleason_logits = _unpack_model_outputs(outputs)
            pirads_loss = _pirads_loss_fn(pirads_logits, labels)
            raw_loss = pirads_loss_weight * pirads_loss

            gleason_loss = torch.zeros((), dtype=labels.dtype, device=labels.device)
            batch_gleason_count = 0
            if gleason_logits is not None:
                saw_gleason_head = True
                gleason_targets = batch_on_device.get(
                    "gleason_target",
                    torch.zeros_like(labels),
                ).to(dtype=labels.dtype)
                gleason_mask = batch_on_device.get(
                    "gleason_mask",
                    torch.zeros_like(labels),
                ).to(dtype=labels.dtype)
                gleason_loss = _gleason_loss_fn(
                    gleason_logits,
                    gleason_targets,
                    gleason_mask,
                )
                raw_loss = raw_loss + gleason_loss_weight * gleason_loss
                batch_mask = gleason_mask > 0.5
                batch_gleason_count = int(batch_mask.sum().item())
                if batch_gleason_count > 0:
                    all_gleason_logits.append(gleason_logits[batch_mask].detach().cpu())
                    all_gleason_labels.append(
                        gleason_targets[batch_mask].detach().cpu()
                    )
                    if batch_accession_ids:
                        all_gleason_accession_ids.extend(
                            [
                                accession_id
                                for accession_id, is_labeled in zip(
                                    batch_accession_ids,
                                    batch_mask.detach().cpu().tolist(),
                                )
                                if is_labeled
                            ]
                        )
                    gleason_loss_total += gleason_loss.item() * batch_gleason_count
                    gleason_label_count += batch_gleason_count

        # Use the true accumulation size for the final (possibly smaller) group to avoid
        # shrinking the gradient on the last optimizer step.
        in_last_group = micro_step > (total_steps - final_accum_steps)
        effective_accum_steps = (
            final_accum_steps if in_last_group else gradient_accumulation_steps
        )
        loss = raw_loss / effective_accum_steps

        loss.backward()

        batch_size = labels.size(0)
        batch_loss = raw_loss.item() * batch_size
        total_loss += batch_loss
        pirads_loss_total += pirads_loss.item() * batch_size
        interval_loss += batch_loss
        interval_samples += batch_size
        all_pirads_logits.append(pirads_logits.detach().cpu())
        all_pirads_labels.append(labels.cpu())
        if batch_accession_ids:
            all_pirads_accession_ids.extend(batch_accession_ids)

        should_step = (micro_step % gradient_accumulation_steps == 0) or (
            micro_step == total_steps
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            global_step = base_step + optimizer_steps

            if log_interval and (optimizer_steps % log_interval == 0):
                avg_interval_loss = interval_loss / max(interval_samples, 1)
                _log_step(optimizer_steps - 1, avg_interval_loss, global_step)
                interval_loss = 0.0
                interval_samples = 0

            if (
                eval_interval_steps
                and eval_callback is not None
                and global_step % eval_interval_steps == 0
            ):
                eval_callback(global_step)
                model.train()

    if log_interval and interval_samples > 0:
        final_step = base_step + optimizer_steps
        avg_interval_loss = interval_loss / interval_samples
        _log_step(optimizer_steps - 1, avg_interval_loss, final_step)

    pirads_logits_cat = torch.cat(all_pirads_logits)
    pirads_labels_cat = torch.cat(all_pirads_labels)
    total_samples = len(pirads_labels_cat)
    avg_loss = total_loss / total_samples
    avg_pirads_loss = pirads_loss_total / total_samples
    avg_gleason_loss = (
        gleason_loss_total / gleason_label_count
        if gleason_label_count > 0
        else float("nan")
    )

    if pbar is not None:
        pbar.close()

    pirads_metrics, _ = _compute_task_metrics(
        logits=pirads_logits_cat,
        targets=pirads_labels_cat,
        accession_ids=all_pirads_accession_ids,
    )
    gleason_logits_cat = (
        torch.cat(all_gleason_logits) if all_gleason_logits else torch.empty(0, dtype=torch.float32)
    )
    gleason_labels_cat = (
        torch.cat(all_gleason_labels) if all_gleason_labels else torch.empty(0, dtype=torch.float32)
    )
    gleason_metrics, aggregated_gleason_count = _compute_task_metrics(
        logits=gleason_logits_cat,
        targets=gleason_labels_cat,
        accession_ids=all_gleason_accession_ids,
    )
    return _merge_task_metrics(
        avg_total_loss=avg_loss,
        avg_pirads_loss=avg_pirads_loss,
        avg_gleason_loss=avg_gleason_loss,
        pirads_metrics=pirads_metrics,
        gleason_metrics=gleason_metrics,
        gleason_labeled_count=aggregated_gleason_count,
        saw_gleason_head=saw_gleason_head,
    )


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    *,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    pirads_loss_weight: float = 1.0,
    gleason_loss_weight: float = 1.0,
    gleason_loss_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    amp_dtype: torch.dtype | None = None,
) -> Dict[str, float]:
    model.eval()
    _pirads_loss_fn = (
        loss_fn if loss_fn is not None else F.binary_cross_entropy_with_logits
    )
    _gleason_loss_fn = (
        gleason_loss_fn if gleason_loss_fn is not None else masked_bce_with_logits
    )
    total_loss = 0.0
    pirads_loss_total = 0.0
    gleason_loss_total = 0.0
    gleason_label_count = 0
    all_pirads_logits = []
    all_pirads_labels = []
    all_pirads_accession_ids: list[str] = []
    all_gleason_logits = []
    all_gleason_labels = []
    all_gleason_accession_ids: list[str] = []
    all_averaging_tags: list[str] = []
    all_gleason_averaging_tags: list[str] = []
    saw_gleason_head = False

    for batch in dataloader:
        labels = batch["label"].to(device)
        batch_accession_ids = list(batch.get("accession_id", []))
        if batch_accession_ids and len(batch_accession_ids) != labels.size(0):
            raise ValueError(
                f"Expected {labels.size(0)} accession IDs in batch, got {len(batch_accession_ids)}"
            )
        batch_averaging_tags = batch.get("averaging_tag")
        if batch_averaging_tags is None:
            averaging_tags: list[str] = []
        elif isinstance(batch_averaging_tags, str):
            averaging_tags = [batch_averaging_tags]
        else:
            averaging_tags = list(batch_averaging_tags)
        if averaging_tags and len(averaging_tags) != labels.size(0):
            raise ValueError(
                f"Expected {labels.size(0)} averaging tags in batch, got {len(averaging_tags)}"
            )
        batch_on_device = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        with _autocast_context(device, amp_dtype):
            outputs = model(batch_on_device)
            pirads_logits, gleason_logits = _unpack_model_outputs(outputs)
            pirads_loss = _pirads_loss_fn(pirads_logits, labels)
            loss = pirads_loss_weight * pirads_loss

            if gleason_logits is not None:
                saw_gleason_head = True
                gleason_targets = batch_on_device.get(
                    "gleason_target",
                    torch.zeros_like(labels),
                ).to(dtype=labels.dtype)
                gleason_mask = batch_on_device.get(
                    "gleason_mask",
                    torch.zeros_like(labels),
                ).to(dtype=labels.dtype)
                gleason_loss = _gleason_loss_fn(
                    gleason_logits,
                    gleason_targets,
                    gleason_mask,
                )
                loss = loss + gleason_loss_weight * gleason_loss
                batch_mask = gleason_mask > 0.5
                batch_gleason_count = int(batch_mask.sum().item())
                if batch_gleason_count > 0:
                    all_gleason_logits.append(gleason_logits[batch_mask].detach().cpu())
                    all_gleason_labels.append(
                        gleason_targets[batch_mask].detach().cpu()
                    )
                    if batch_accession_ids:
                        all_gleason_accession_ids.extend(
                            [
                                accession_id
                                for accession_id, is_labeled in zip(
                                    batch_accession_ids,
                                    batch_mask.detach().cpu().tolist(),
                                )
                                if is_labeled
                            ]
                        )
                    gleason_loss_total += gleason_loss.item() * batch_gleason_count
                    gleason_label_count += batch_gleason_count
                    if averaging_tags:
                        all_gleason_averaging_tags.extend(
                            [
                                tag
                                for tag, is_labeled in zip(
                                    averaging_tags,
                                    batch_mask.detach().cpu().tolist(),
                                )
                                if is_labeled
                            ]
                        )

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        pirads_loss_total += pirads_loss.item() * batch_size
        all_pirads_logits.append(pirads_logits.detach().cpu())
        all_pirads_labels.append(labels.cpu())
        if batch_accession_ids:
            all_pirads_accession_ids.extend(batch_accession_ids)
        if averaging_tags:
            all_averaging_tags.extend(averaging_tags)

    pirads_logits_cat = torch.cat(all_pirads_logits)
    pirads_labels_cat = torch.cat(all_pirads_labels)
    total_samples = len(pirads_labels_cat)
    avg_loss = total_loss / total_samples
    avg_pirads_loss = pirads_loss_total / total_samples
    avg_gleason_loss = (
        gleason_loss_total / gleason_label_count
        if gleason_label_count > 0
        else float("nan")
    )

    pirads_metrics, _ = _compute_task_metrics(
        logits=pirads_logits_cat,
        targets=pirads_labels_cat,
        accession_ids=all_pirads_accession_ids,
    )
    gleason_logits_cat = (
        torch.cat(all_gleason_logits) if all_gleason_logits else torch.empty(0, dtype=torch.float32)
    )
    gleason_labels_cat = (
        torch.cat(all_gleason_labels) if all_gleason_labels else torch.empty(0, dtype=torch.float32)
    )
    gleason_metrics, aggregated_gleason_count = _compute_task_metrics(
        logits=gleason_logits_cat,
        targets=gleason_labels_cat,
        accession_ids=all_gleason_accession_ids,
    )
    metrics = _merge_task_metrics(
        avg_total_loss=avg_loss,
        avg_pirads_loss=avg_pirads_loss,
        avg_gleason_loss=avg_gleason_loss,
        pirads_metrics=pirads_metrics,
        gleason_metrics=gleason_metrics,
        gleason_labeled_count=aggregated_gleason_count,
        saw_gleason_head=saw_gleason_head,
    )
    if all_averaging_tags:
        metrics.update(
            _group_metrics_by_averaging_tag(
                task_name="pirads",
                logits=pirads_logits_cat,
                targets=pirads_labels_cat,
                averaging_tags=all_averaging_tags,
                count_key_suffix="count",
            )
        )
    if saw_gleason_head and all_gleason_averaging_tags:
        metrics.update(
            _group_metrics_by_averaging_tag(
                task_name="gleason",
                logits=gleason_logits_cat,
                targets=gleason_labels_cat,
                averaging_tags=all_gleason_averaging_tags,
                count_key_suffix="labeled_count",
            )
        )
    return metrics
