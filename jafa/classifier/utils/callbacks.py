import lightning as pl
from lightning.pytorch.callbacks import Callback
import wandb

from classifier.utils import transforms


class LogGenerationCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        if batch_idx % 32 == 0:
            recon_dict = pl_module.reconstruct_image(batch)
            recon_images = recon_dict["output"]
            target_images = recon_dict["target"]

            trainer.logger.experiment.log(
                {
                    "outputs": [wandb.Image(x) for x in recon_images],
                    "targets": [wandb.Image(x) for x in target_images],
                    "global_step": trainer.global_step,
                }
            )


class LogMaskCallback(Callback):
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx == 0:
            # Log only for image-like inputs to avoid shape mismatches on k-space
            if getattr(pl_module, "input_type", "kspace") != "image":
                return
            image = batch.get("image", batch.get("recon_esc"))
            if image is None:
                return
            # Ensure 4D (B, C, H, W)
            if image.dim() == 2:
                image = image.unsqueeze(0).unsqueeze(0)
            elif image.dim() == 3:
                image = image.unsqueeze(1)
            try:
                masked_x = pl_module.mask_fn(image, batch_size=image.shape[0])
            except TypeError:
                masked_x = pl_module.mask_fn(image)
            if "vol" not in pl_module.config.dataset and trainer.logger is not None:
                mask_img = masked_x[0].cpu().numpy()
                if mask_img.ndim == 3:
                    mask_img = mask_img.transpose(1, 2, 0)
                trainer.logger.experiment.log(
                    {
                        "mask": [wandb.Image(mask_img)],
                        "global_step": trainer.global_step,
                    }
                )
