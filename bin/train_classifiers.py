import argparse
import os
import sys
import click

import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightning as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger

from classifier.classification_modules.modules import ARMSModule, BayesDistillModule

from torch.utils.data import DataLoader
from classifier.data import (
    MRDataModule,
    kspaceCIFAR10,
    ImageNetDataset,
    HierarchicalImbalancedImageNet,
    kspaceImageNet,
)
from classifier.utils.utils import get_config
from classifier.utils.callbacks import LogMaskCallback


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, config):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False,
        )


def get_data(config: argparse.Namespace, dev_mode: bool) -> pl.LightningDataModule:
    print("Using workers -> ", config.num_workers)
    if config.dataset in ["cifar10", "imagenet"]:
        input_type = getattr(config, "input_type", "image")
        if config.dataset == "cifar10":
            train_dataset = kspaceCIFAR10(config, mode="train")
            val_dataset = kspaceCIFAR10(config, mode="val")
            test_dataset = kspaceCIFAR10(config, mode="test")
        else:
            if (
                getattr(config, "hierarchical_superclass", None) is not None
                and getattr(config, "hierarchical_target_class", None) is not None
            ):
                train_dataset = HierarchicalImbalancedImageNet(config, mode="train")
                val_dataset = HierarchicalImbalancedImageNet(config, mode="val")
                test_dataset = HierarchicalImbalancedImageNet(config, mode="test")
            else:
                dataset_cls = kspaceImageNet if input_type == "kspace" else ImageNetDataset
                train_dataset = dataset_cls(config, mode="train")
                val_dataset = dataset_cls(config, mode="val")
                test_dataset = dataset_cls(config, mode="test")
        return SimpleDataModule(train_dataset, val_dataset, test_dataset, config)

    return MRDataModule(config=config, dev_mode=dev_mode)


def get_model(config: argparse.ArgumentParser):
    """
    Get model based on classifier_type.
    Supports:
    - "arms": ARMSModule
    - "bayes_distill_arms": BayesDistillModule
    - "bayes_distill": BayesDistillModule
    """
    # Add compatibility for BayesDistillClassifier config parameters
    # Map lr -> learning_rate and ckpt_dir -> checkpoint_dir if needed
    classifier_type = getattr(config, "classifier_type", "arms")
    if classifier_type in ["bayes_distill", "bayes_distill_arms"]:
        # Map lr to learning_rate if learning_rate is not set
        if not hasattr(config, 'learning_rate') and hasattr(config, 'lr'):
            config.learning_rate = config.lr
        # Map ckpt_dir to checkpoint_dir if checkpoint_dir is not set
        if not hasattr(config, 'checkpoint_dir') and hasattr(config, 'ckpt_dir'):
            config.checkpoint_dir = config.ckpt_dir
    
    if classifier_type in ["arms", "fully_sampled"]:
        model = ARMSModule(config=config)
    elif classifier_type in ["bayes_distill", "bayes_distill_arms"]:
        model = BayesDistillModule(config=config)
    else:
        raise NotImplementedError(
            f"classifier type {config.classifier_type} not implemented"
        )

    if getattr(config, "resume_train", False):
        print(f"Loading ckpt from {config.ckpt_path}")
        model = model.load_from_checkpoint(config.ckpt_path)
        resume_from_checkpoint = config.ckpt_path
    else:
        resume_from_checkpoint = None

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    return model, resume_from_checkpoint


def train_model(
    config: argparse.Namespace,
    datamodule: pl.LightningDataModule,
    offline: bool,
) -> pl.LightningModule:

    print(config)
    # torch.set_float32_matmul_precision('medium')

    project_name = (
        config.project_name
        if config.project_name
        else f"classifier_{config.dataset}"
    )

    ckpt_dir = getattr(config, "ckpt_dir", getattr(config, "checkpoint_dir", "./checkpoints"))
    wandb_logger = WandbLogger(
        project=project_name,
        save_dir=ckpt_dir,
        log_model=True if not offline else False,
        offline=offline,
    )

    classifier_type = getattr(config, "classifier_type", "arms")
    checkpoint_filename = (
        f"{classifier_type}-{config.dataset}-{config.model_type}-"
        + "{epoch}-{best_metric:.2f}"
    )

    model, resume_from_checkpoint = get_model(config)
    
    # checkpoint callback
    model_checkpoint = ModelCheckpoint(
        monitor="best_metric",
        filename=checkpoint_filename,
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # mask monitoring
    mask_callback = LogMaskCallback()

    # early_stop_callback = EarlyStopping(monitor='best_metric', patience=5, mode='max', log_rank_zero_only=True)


    max_epochs = getattr(config, "n_epochs", getattr(config, "max_epochs", 100))
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=max_epochs,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[model_checkpoint, lr_monitor],
        # callbacks=[mask_callback, model_checkpoint, early_stop_callback, lr_monitor],
        gradient_clip_val=getattr(config, "grad_clip_val", 0.0),
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=10,
        accumulate_grad_batches=getattr(config, "accumulate_grad_batches", 1),
    )
    # torch.use_deterministic_algorithms(True, warn_only=True)

    # train the models
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)

    # test the models : disabling testing for now, validation is still performed in train loop
    # trainer.test(model, datamodule, ckpt_path=resume_from_checkpoint)


@click.command()
@click.option("--config_path", required=True, help="Config file path.")
@click.option(
    "--offline",
    type=bool,
    default=False,
    required=False,
    help="run wandb in offline mode.",
)
@click.option(
    "--dev_mode",
    type=bool,
    default=False,
    required=False,
    help="run code in debug mode.",
)
def main(config_path, offline, dev_mode):
    config = get_config(config_path)
    datamodule = get_data(config=config, dev_mode=dev_mode)

    if dev_mode:
        config.use_weighted_sampler = False

    train_model(config=config, datamodule=datamodule, offline=offline)


if __name__ == "__main__":
    main()
