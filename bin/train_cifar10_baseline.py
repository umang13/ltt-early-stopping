#!/usr/bin/env python3
"""
Training script for baseline CIFAR-10 classifier (no Bayes distilling)

This script trains a standard classifier on CIFAR-10 to establish baseline accuracy.

Usage:
    python bin/train_cifar10_baseline.py --config configs/cifar10_baseline_config.yaml
    python bin/train_cifar10_baseline.py --config configs/cifar10_baseline_config.yaml --max_epochs 100
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader

from classifier.data import kspaceCIFAR10
from classifier.classification_modules.modules import ARMSModule
from classifier.utils import get_config


def main():
    parser = argparse.ArgumentParser(description='Train baseline CIFAR-10 classifier')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of epochs (overrides config)')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16,
                       help='Precision (16 or 32)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--validation_only', action='store_true',
                       help='Only run validation epoch (requires checkpoint)')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments if provided
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    
    print("="*80)
    print("BASELINE CIFAR-10 CLASSIFIER TRAINING")
    print("="*80)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model_type}")
    print(f"Pretrained: {config.pretrained}")
    print(f"Input type: {getattr(config, 'input_type', 'image')}")
    print(f"Mask type: {getattr(config, 'mask_type', 'none')}")
    print(f"Max epochs: {config.max_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Dropout rate: {config.dropout}")
    print("="*80)
    
    # Set up data
    train_dataset = kspaceCIFAR10(config, mode="train")
    val_dataset = kspaceCIFAR10(config, mode="val")
    test_dataset = kspaceCIFAR10(config, mode="test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create model
    model = ARMSModule(config)
    
    print(f"\nModel created: {model.__class__.__name__}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup callbacks
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="cifar10_baseline_{epoch:02d}_{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=getattr(config, 'early_stopping_patience', 15),
        mode="max",
        verbose=True
    )
    
    # Setup logger
    if not args.disable_wandb:
        wandb_logger = WandbLogger(
            project=config.project_name,
            name=config.run_name,
            save_dir=config.checkpoint_dir
        )
    else:
        wandb_logger = None
        print("Weights & Biases logging disabled")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        log_every_n_steps=10,
    )
    
    # Train or validate
    if args.validation_only:
        print("\n" + "="*80)
        print("VALIDATION ONLY MODE")
        print("="*80)
        
        if args.resume_from_checkpoint:
            print(f"Loading checkpoint: {args.resume_from_checkpoint}")
            model = ARMSModule.load_from_checkpoint(args.resume_from_checkpoint, config=config)
        else:
            print("Warning: No checkpoint specified for validation-only mode. Using untrained model.")
        
        trainer.validate(model, val_loader)
        print("\nValidation completed!")
    else:
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        # Train the model
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=args.resume_from_checkpoint
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Best val_cifar10_accuracy: {checkpoint_callback.best_model_score:.4f}")
        
        # Test on test set
        print("\n" + "="*80)
        print("TESTING ON TEST SET")
        print("="*80)
        
        # Load best checkpoint for testing
        best_model = ARMSModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            config=config
        )
        
        test_results = trainer.test(best_model, test_loader)
        
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
        if test_results:
            test_acc = test_results[0].get('test_accuracy', 0)
            print(f"Test accuracy: {test_acc:.4f}")
        print(f"Checkpoints saved to: {config.checkpoint_dir}")
        print(f"Best model: {checkpoint_callback.best_model_path}")
        print("="*80)


if __name__ == "__main__":
    main()

