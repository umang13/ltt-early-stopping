import argparse
from copy import deepcopy

import lightning as pl
import torch

from classifier.metrics.classification_metrics import evaluate_classifier

from classifier.masks.factory import build_mask_fn
from classifier.classification_modules.models.factory import build_model
from classifier.classification_modules.modules.common import (
    compute_accuracy,
    compute_kl_divergence,
    get_normalization_fn,
    get_sampling_rate,
    init_loss_functions,
    resolve_label_names,
    resolve_num_classes,
)


class BayesDistillModule(pl.LightningModule):
    """
    Unified Bayes distillation module:
    - cross-entropy on fully sampled input
    - KL divergence between full and masked logits
    """

    def __init__(self, config: argparse.Namespace):
        super().__init__()
        config_dict = vars(config) if hasattr(config, "__dict__") else dict(config)
        self.save_hyperparameters(config_dict)
        self.config = config

        self.input_type = getattr(config, "input_type", "kspace")
        # Single-coil only: always use sc_kspace
        self.label_names = resolve_label_names(config)
        self.num_classes, self.is_imbalanced = resolve_num_classes(config)

        self.kl_weight = getattr(config, "kl_weight", 1.0)
        self.adaptive_kl_weight = getattr(config, "adaptive_kl_weight", False)
        self.kl_weight_min = getattr(config, "kl_weight_min", 0.1)
        self.kl_weight_max = getattr(config, "kl_weight_max", 2.0)
        self.kl_weight_power = getattr(config, "kl_weight_power", 1.0)

        self.model = build_model(config)
        self.mask_fn = build_mask_fn(config, training=self.training)
        self.normalization_fn = get_normalization_fn(config)
        self.criterion = init_loss_functions(config, self.label_names, self.num_classes)
        self.val_criterion = init_loss_functions(
            config, self.label_names, self.num_classes, for_validation=True
        )

        self.best_metric = 0.0
        self.validation_step_outputs = []

    def on_fit_start(self) -> None:
        if hasattr(self.mask_fn, "device"):
            self.mask_fn.device = self.device
        self.val_criterion = init_loss_functions(
            self.config, self.label_names, self.num_classes, for_validation=True
        )
        return super().on_fit_start()

    def _get_input(self, batch, masked=True, fixed_mask=None, mask_fn=None):
        if mask_fn is None:
            mask_fn = self.mask_fn
        if self.input_type == "kspace":
            kspace = batch["sc_kspace"]
            if not masked:
                return kspace
            if fixed_mask is not None:
                return kspace * fixed_mask
            if "vol" in self.config.dataset and hasattr(mask_fn, "__call__"):
                num_slices = kspace.shape[1]
                return mask_fn(kspace, num_slices=num_slices)
            return mask_fn(kspace)

        image = batch.get("recon_esc", batch.get("image"))
        image = self.normalization_fn(image)
        if masked:
            try:
                image = mask_fn(image, batch_size=image.shape[0])
            except TypeError:
                image = mask_fn(image)
        return image

    def forward_fully_sampled(self, batch):
        inputs = self._get_input(batch, masked=False)
        return self.model(inputs)

    def forward_undersampled(self, batch, fixed_mask=None, mask_fn=None):
        # input_tensor = None
        # if self.input_type == "kspace":
        #     input_tensor = batch["sc_kspace"]
        # else:
        #     input_tensor = batch.get("recon_esc", batch.get("image"))
        # if input_tensor is not None :
        #     num_zeros_before = (input_tensor == 0).sum().item()
        #     total_before = input_tensor.numel()
        #     pct_before = 100.0 * num_zeros_before / max(total_before, 1)
        #     print(f"[BayesDistill] zeros before mask: {num_zeros_before}/{total_before} ({pct_before:.2f}%)")
        inputs = self._get_input(batch, masked=True, fixed_mask=fixed_mask, mask_fn=mask_fn)
        # if inputs is not None:
        #     num_zeros_after = (inputs == 0).sum().item()
        #     total_after = inputs.numel()
        #     pct_after = 100.0 * num_zeros_after / max(total_after, 1)
        #     print(f"[BayesDistill] zeros after mask: {num_zeros_after}/{total_after} ({pct_after:.2f}%)")
        return self.model(inputs)

    def _get_val_mask_fn(self):
        val_sampling_ratio = getattr(self.config, "val_sampling_ratio", 0.1)
        cfg = deepcopy(self.config)

        if cfg.mask_type == "random_rate_patch":
            mask_ratio = 1.0 - val_sampling_ratio
            cfg.min_sampling_rate = mask_ratio
            cfg.max_sampling_rate = mask_ratio
        elif cfg.mask_type in ["random_rate_vds", "random_rate_uniform"]:
            cfg.min_sampling_rate = val_sampling_ratio
            cfg.max_sampling_rate = val_sampling_ratio
        elif cfg.mask_type in ["random_vds", "random_vvds"]:
            cfg.k_fraction = val_sampling_ratio
            cfg.min_k_fraction = val_sampling_ratio
            cfg.max_k_fraction = val_sampling_ratio

        val_mask_fn = build_mask_fn(cfg, training=False)
        if hasattr(val_mask_fn, "device"):
            val_mask_fn.device = self.device
        return val_mask_fn

    def _compute_adaptive_kl_weight(self):
        if not self.adaptive_kl_weight:
            return self.kl_weight
        sampling_rate = get_sampling_rate(self.mask_fn)
        if sampling_rate is None:
            return self.kl_weight
        masking_rate = 1.0 - sampling_rate
        scaled = masking_rate ** self.kl_weight_power
        return self.kl_weight_min + (self.kl_weight_max - self.kl_weight_min) * scaled

    def training_step(self, batch, batch_idx):
        full_logits = self.forward_fully_sampled(batch)
        masked_logits = self.forward_undersampled(batch)

        total_loss = 0.0
        total_acc = 0.0
        kl_weight = self._compute_adaptive_kl_weight()

        for label_name in self.label_names:
            labels = batch["label"][label_name].squeeze(1)
            ce_loss = self.criterion[label_name](full_logits[label_name], labels)
            kl_loss = compute_kl_divergence(full_logits[label_name], masked_logits[label_name])
            total_loss += ce_loss + kl_weight * kl_loss
            total_acc += compute_accuracy(masked_logits[label_name], labels)

        avg_acc = total_acc / len(self.label_names)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", avg_acc, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        val_mask_fn = self._get_val_mask_fn()
        masked_logits = self.forward_undersampled(batch, mask_fn=val_mask_fn)
        total_loss = 0.0
        total_acc = 0.0
        for label_name in self.label_names:
            labels = batch["label"][label_name].squeeze(1)
            loss = self.val_criterion[label_name](masked_logits[label_name], labels)
            acc = compute_accuracy(masked_logits[label_name], labels)
            total_loss += loss
            total_acc += acc

        avg_acc = total_acc / len(self.label_names)
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", avg_acc, on_epoch=True, prog_bar=True)
        step_output = {}
        for label_name in self.label_names:
            step_output[f"preds_{label_name}"] = masked_logits[label_name].detach()
            step_output[f"labels_{label_name}"] = batch["label"][label_name].detach().squeeze(1)
        step_output["preds"] = masked_logits
        step_output["labels"] = batch["label"]
        step_output["loss"] = total_loss
        self.validation_step_outputs.append(step_output)
        return step_output

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = {label_name: [] for label_name in self.label_names}
        all_labels = {label_name: [] for label_name in self.label_names}

        for step_output in self.validation_step_outputs:
            for label_name in self.label_names:
                all_preds[label_name].append(step_output[f"preds_{label_name}"])
                all_labels[label_name].append(step_output[f"labels_{label_name}"])

        avg_auc = 0.0
        for label_name in self.label_names:
            preds = torch.cat(all_preds[label_name], dim=0)
            labels = torch.cat(all_labels[label_name], dim=0)
            metrics = evaluate_classifier(
                preds=preds,
                labels=labels,
                num_classes=self.num_classes,
            )
            self.log(f"val_{label_name}_auc", metrics["auc"], on_epoch=True, prog_bar=True)
            self.log(
                f"val_{label_name}_balanced_accuracy",
                metrics["balanced_accuracy"],
                on_epoch=True,
                prog_bar=True,
            )
            avg_auc += metrics["auc"] / len(self.label_names)

        if avg_auc > self.best_metric:
            self.best_metric = avg_auc
        self.log("val_auc_mean", avg_auc, on_epoch=True, prog_bar=True)
        self.log("best_metric", self.best_metric, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        masked_logits = self.forward_undersampled(batch)
        total_acc = 0.0
        for label_name in self.label_names:
            labels = batch["label"][label_name].squeeze(1)
            total_acc += compute_accuracy(masked_logits[label_name], labels)
        avg_acc = total_acc / len(self.label_names)
        self.log("test_accuracy", avg_acc, on_epoch=True, prog_bar=True)
        return avg_acc

    def configure_optimizers(self):
        learning_rate = getattr(self.config, "learning_rate", getattr(self.config, "lr", 1e-3))
        parameters = list(self.model.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=getattr(self.config, "weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self.config.early_stopping_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            },
        }
