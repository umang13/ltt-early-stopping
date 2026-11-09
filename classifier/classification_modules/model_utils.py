"""
Utility functions for model loading and backbone extraction.
"""

import argparse
import torch
import torch.nn as nn
from copy import deepcopy
import pytorch_lightning as pl

from classifier.utils.utils import get_config
from classifier.data import MRDataModule

from .arms_models import kSpaceNet, kspaceMRNet
from .compat import load_state_dict_with_classifier_fallback, infer_pool_type_from_state_dict


def get_ckpt_backbone(config, path, classifier_class=None):
    """
    Generic function to extract backbone from a checkpoint.
    
    This function can be used with any classifier class that follows the standard
    structure (ARMSClassifier, BayesDistillArmsClassifier, Indirect_MRClassifier, etc.)
    
    Args:
        config: Configuration object
        path: Path to checkpoint file
        classifier_class: The classifier class to use for loading the checkpoint.
                        If None, will try to infer from config or use ARMSClassifier as default.
    
    Returns:
        backbone: The extracted backbone model
    """
    # Store original model_type to restore later
    # original_model_type = config.model_type
    # config.model_type = config.slice_model_type
    if config.slice_config_path is not None:
        slice_config = get_config(config.slice_config_path)
    else:
        slice_config = config

    # If classifier_class is not provided, try to import ARMSClassifier as default
    # Use lazy import to avoid circular dependency
    if classifier_class is None:
        from classifier.classification_modules import ARMSClassifier
        classifier_class = ARMSClassifier

    # Load model from checkpoint
    remap_classifier_keys = False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        inferred_pool = infer_pool_type_from_state_dict(
            state_dict, getattr(slice_config, "pool_type", None)
        )
        if inferred_pool is not None:
            slice_config.pool_type = inferred_pool
        has_classifier_keys = any(k.startswith("classifier.") for k in state_dict.keys())
        has_model_keys = any(k.startswith("model.") for k in state_dict.keys())
        remap_classifier_keys = has_classifier_keys and not has_model_keys
    except Exception:
        pass
    try:
        if remap_classifier_keys:
            print(
                "Skipping load_from_checkpoint (classifier.* keys detected). "
                "Using manual load with remap."
            )
            loaded_classifier = classifier_class(config=slice_config)
        else:
            loaded_classifier = classifier_class.load_from_checkpoint(path, config=slice_config)
    except Exception as e:
        print(f"load_from_checkpoint failed, falling back to manual load: {e}")
        loaded_classifier = classifier_class(config=slice_config)
    try:
        missing, unexpected = load_state_dict_with_classifier_fallback(
            loaded_classifier, path, map_location="cpu", strict=False, weights_only=False
        )
        print(f"Loaded slice checkpoint from {path}")
        if missing or unexpected:
            print(f"Remapped classifier->model. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    except Exception as e:
        print(f"Warning: could not manually load state_dict: {e}")
        exit(1)
    
    # Perform validation on slice-level dataset
    # print("\n" + "="*80)
    # print("Running validation on slice-level dataset")
    # print("="*80)
    # try:
    #     # Initialize slice-level datamodule
    #     slice_datamodule = MRDataModule(config=slice_config, dev_mode=False)
    #     slice_datamodule.setup("validate")
        
    #     # Create a minimal trainer for validation
    #     val_trainer = pl.Trainer(
    #         devices="auto",
    #         accelerator="auto",
    #         logger=False,  # Don't log to avoid conflicts
    #         enable_progress_bar=True,
    #     )
        
    #     # Run validation
    #     val_metrics = val_trainer.validate(loaded_classifier, slice_datamodule)
        
    #     # Print validation metrics
    #     print("\n" + "-"*80)
    #     print("Slice-level validation metrics:")
    #     print("-"*80)
    #     if val_metrics and len(val_metrics) > 0:
    #         for key, value in val_metrics[0].items():
    #             print(f"  {key}: {value}")
    #     else:
    #         print("  No metrics returned from validation")
    #     print("-"*80 + "\n")
    # except Exception as e:
    #     print(f"Warning: Could not run slice-level validation: {e}")
    #     print("Continuing with backbone extraction...")
    #     import traceback
    #     traceback.print_exc()
    
    # Handle different attribute names: ARMSClassifier uses .model, BayesDistillArmsClassifier uses .classifier
    if hasattr(loaded_classifier, 'model'):
        model = loaded_classifier.model
        
        # Print all module names
        print("\n" + "="*80)
        print("All modules in loaded_classifier.model:")
        print("="*80)
        for name, module in model.named_modules():
            print(f"  {name}: {type(module).__name__}")
        print("="*80 + "\n")
        
        # model.conv_kspace = loaded_classifier.model.conv_kspace
    elif hasattr(loaded_classifier, 'classifier'):
        model = loaded_classifier.classifier
        model.conv_kspace = loaded_classifier.classifier.conv_kspace
        # model.layernorm = loaded_classifier.classifier.layernorm
    else:
        raise AttributeError(
            f"Classifier {classifier_class.__name__} does not have 'model' or 'classifier' attribute. "
            f"Available attributes: {[attr for attr in dir(loaded_classifier) if not attr.startswith('_')]}"
        )

    assert (
        model.config.model_type == config.slice_model_type
    ), f"Model type {model.config.model_type} does not match slice model type {config.slice_model_type}"

    # Extract backbone based on slice_model_type
    # Note: model.backbone is already a Sequential that excludes the final fc layer
    if config.slice_model_type == "alexnet":
        backbone = model.features
    elif config.slice_model_type == "resnet50":
        # model.backbone is already a Sequential with [:-1], use it directly
        backbone = model.backbone
    elif config.slice_model_type == "resnet18":
        # NOTE: Creating a new Sequential here - we'll verify weights are preserved
        # The children are references, so weights should be shared, but we'll verify
        backbone = nn.Sequential(*list(model.backbone.children()))
        print(f"WARNING: Created new Sequential for resnet18 backbone. Will verify weights are preserved.")
    else:
        raise NotImplementedError(
            f"Slice model {config.slice_model_type} not implemented"
        )
    
    # Store weight statistics from loaded backbone for verification
    print("\n" + "="*80)
    print("Storing weight statistics from loaded slice classifier backbone")
    print("="*80)
    loaded_backbone_weights = {}
    for name, param in model.backbone.named_parameters():
        loaded_backbone_weights[name] = {
            'tensor': param.data.clone(),  # Clone to preserve original
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'min': param.data.min().item(),
            'max': param.data.max().item(),
        }
        if len(loaded_backbone_weights) <= 3:  # Print first 3 for verification
            print(f"  {name}: shape={param.shape}, mean={loaded_backbone_weights[name]['mean']:.6f}, std={loaded_backbone_weights[name]['std']:.6f}")
    print(f"Total parameters stored: {len(loaded_backbone_weights)}")
    print("="*80 + "\n")
    
    # Store conv_kspace weight statistics
    loaded_conv_kspace_weight = {
        'tensor': model.conv_kspace.weight.data.clone(),
        'mean': model.conv_kspace.weight.data.abs().mean().item(),
        'std': model.conv_kspace.weight.data.abs().std().item(),
        'min': model.conv_kspace.weight.data.abs().min().item(),
        'max': model.conv_kspace.weight.data.abs().max().item(),
    }
    print(f"Loaded conv_kspace weight - mean: {loaded_conv_kspace_weight['mean']:.6f}, std: {loaded_conv_kspace_weight['std']:.6f}, min: {loaded_conv_kspace_weight['min']:.6f}, max: {loaded_conv_kspace_weight['max']:.6f}")
    
    # Restore original model_type
    # config.model_type = original_model_type
    print("Conv kspace weight at loaded checkpoint: ", model.conv_kspace.weight.data.abs().mean().item())
    conv_kspace = model.conv_kspace
    
    # Extract slice-level label layers for slice-level predictions
    slice_label_layers = model.label_layers
    slice_label_names = model.label_names
    print(f"\nExtracted slice-level label_layers: {len(slice_label_layers)} layers")
    for i, layer in enumerate(slice_label_layers):
        label_name = slice_label_names[i] if i < len(slice_label_names) else f"label_{i}"
        print(f"  Label layer {i} ({label_name}): {layer}")
    
    # Return backbone, conv_kspace, slice_label_layers, slice_label_names, and weight references for verification
    return backbone, conv_kspace, slice_label_layers, slice_label_names, loaded_backbone_weights, loaded_conv_kspace_weight


def get_model(config, classifier_class=None) -> nn.Module:
    """
    Get model based on config.model_type.
    
    Creates kSpaceNet for standard model types (resnet18, resnet50, alexnet, vit_b_16)
    or kspaceMRNet for mrnet type. Optionally loads pretrained backbone from slice classifier.
    
    Args:
        config: Configuration object with model_type, pretrained, slice_classifier_ckpt, etc.
        classifier_class: The classifier class to use for loading the backbone checkpoint.
                        If None, defaults to ARMSClassifier in get_ckpt_backbone.
    
    Returns:
        model: The initialized model (kSpaceNet or kspaceMRNet)
    """
    if config.model_type in ["resnet18","resnet50", "alexnet", "vit_b_16"]:
        model = kSpaceNet(config=config)

    elif config.model_type == "mrnet":
        model = kspaceMRNet(config=config)

        if config.pretrained == "slice_classifier":
            print("*" * 100)
            print("DEBUG: Loading ckpt from slice classifier")
            print("*" * 100)
            # return model
    
            slice_config = get_config(config.slice_config_path)
            if slice_config.classifier_type == "bayes_distill_arms":
                from classifier.classification_modules import BayesDistillArmsClassifier
                classifier_class = BayesDistillArmsClassifier
            else:
                from classifier.classification_modules import ARMSClassifier
                classifier_class = ARMSClassifier
            # Pass classifier_class to get_ckpt_backbone (defaults to ARMSClassifier if None)
            backbone, conv_kspace, slice_label_layers, slice_label_names, loaded_backbone_weights, loaded_conv_kspace_weight = get_ckpt_backbone(
                config=config, path=config.slice_classifier_ckpt, classifier_class=classifier_class
            )
            # loaded_classifier = get_ckpt_backbone(
            #     config=config, path=config.slice_classifier_ckpt, classifier_class=classifier_class
            # )
            model.backbone = backbone
            # model.backbone.load_state_dict(loaded_classifier.classifier.backbone.state_dict())
            
            model.conv_kspace = conv_kspace
            # model.conv_kspace.load_state_dict(loaded_classifier.classifier.conv_kspace.state_dict())
            # model.conv_kspace.load_state_dict(loaded_classifier.classifier.conv_kspace.state_dict())
            print("Conv kspace weight at volume model: ", model.conv_kspace.weight.data.abs().mean().item())
            # Initialize label layers with the same weights as the slice classifier
            # model.label_layers.load_state_dict(loaded_classifier.classifier.label_layers.state_dict())
            # # model.label_layers.load_state_dict(loaded_classifier.classifier.label_layers.state_dict())
            # for p in model.label_layers.parameters():
            #     p.requires_grad = True
            # Store slice-level label layers for slice-level predictions
            for i in range(0, config.num_labels):
                # model.label_layers.append(nn.Linear(512, config.num_classes))
                model.label_layers[i].load_state_dict(slice_label_layers[i].state_dict())
            # model.label_layers.train()
            model.slice_label_layers = slice_label_layers
            model.slice_label_names = slice_label_names
            # print(f"Stored slice-level label layers in volume model. Slice label names: {model.slice_label_names}, Volume label names: {model.label_names}")
            
            # # VERIFICATION: Ensure weights are actually transferred correctly
            # print("\n" + "="*80)
            # print("VERIFICATION: Checking that weights are correctly loaded in volume classifier")
            # print("="*80)
            
            # # Verify backbone weights
            # all_weights_match = True
            # exact_matches = 0
            # close_matches = 0
            # mismatches = 0
            # missing = 0
            
            # for name, loaded_stats in loaded_backbone_weights.items():
            #     if name in dict(model.backbone.named_parameters()):
            #         current_param = dict(model.backbone.named_parameters())[name]
            #         # Check if tensors are identical (exact match)
            #         if torch.equal(current_param.data, loaded_stats['tensor']):
            #             exact_matches += 1
            #             if exact_matches <= 3:  # Print first 3 exact matches
            #                 print(f"  ✓ {name}: weights match EXACTLY")
            #         else:
            #             # Check if they're close (within numerical precision)
            #             max_diff = (current_param.data - loaded_stats['tensor']).abs().max().item()
            #             if max_diff > 1e-6:
            #                 mismatches += 1
            #                 print(f"  ✗ {name}: weights DO NOT MATCH! Max difference: {max_diff:.2e}")
            #                 all_weights_match = False
            #             else:
            #                 close_matches += 1
            #                 if close_matches <= 3:  # Print first 3 close matches
            #                     print(f"  ✓ {name}: weights match (within numerical precision, max_diff={max_diff:.2e})")
            #     else:
            #         missing += 1
            #         print(f"  ✗ {name}: not found in volume classifier backbone!")
            #         all_weights_match = False
            
            # print(f"\n  Backbone weight verification summary:")
            # print(f"    Exact matches: {exact_matches}/{len(loaded_backbone_weights)}")
            # print(f"    Close matches (within 1e-6): {close_matches}/{len(loaded_backbone_weights)}")
            # print(f"    Mismatches: {mismatches}/{len(loaded_backbone_weights)}")
            # print(f"    Missing: {missing}/{len(loaded_backbone_weights)}")
            
            # # Verify conv_kspace weights
            # if torch.equal(model.conv_kspace.weight.data, loaded_conv_kspace_weight['tensor']):
            #     print(f"  ✓ conv_kspace: weights match EXACTLY")
            # else:
            #     max_diff = (model.conv_kspace.weight.data - loaded_conv_kspace_weight['tensor']).abs().max().item()
            #     if max_diff > 1e-6:
            #         print(f"  WARNING: conv_kspace weights DO NOT MATCH! Max difference: {max_diff:.2e}")
            #         all_weights_match = False
            #     else:
            #         print(f"  ✓ conv_kspace: weights match (within numerical precision, max_diff={max_diff:.2e})")
            
            # if all_weights_match:
            #     print("\n  ✓✓✓ ALL WEIGHTS VERIFIED: Volume classifier is using loaded slice classifier weights ✓✓✓")
            # else:
            #     print("\n  ✗✗✗ WARNING: Some weights do not match! The volume classifier may not be using the loaded weights correctly! ✗✗✗")
            
            # print("="*80 + "\n")
            # # Debug: Verify pretrained weights are loaded
            # debug = getattr(config, 'debug_forward', False)
            # if debug:
            #     print("\n" + "="*80)
            #     print("DEBUG: Verifying pretrained weights are loaded")
            #     print("="*80)
                
            #     # Check backbone structure
            #     print(f"DEBUG: Backbone type: {type(backbone)}")
            #     if isinstance(backbone, nn.Sequential):
            #         print(f"DEBUG: Backbone has {len(list(backbone.children()))} children")
            #         print(f"DEBUG: First few backbone layers: {list(backbone.children())[:3]}")
                
            #     # Check conv_kspace parameters
            #     print(f"DEBUG: conv_kspace kernel_size: {conv_kspace.kernel_size}")
            #     print(f"DEBUG: conv_kspace in_channels: {conv_kspace.in_channels}, out_channels: {conv_kspace.out_channels}")
                
            #     # Check if weights are actually different from random initialization
            #     # Get a sample weight from backbone
            #     for name, param in list(backbone.named_parameters())[:1]:
            #         print(f"DEBUG: Sample backbone param '{name}' shape: {param.shape}, mean: {param.mean():.6f}, std: {param.std():.6f}")
            #         break
                
            #     # Get a sample weight from conv_kspace
            #     print(f"DEBUG: conv_kspace weight shape: {conv_kspace.weight.shape}")
            #     print(f"DEBUG: conv_kspace weight real - min: {conv_kspace.weight.real.min():.6f}, max: {conv_kspace.weight.real.max():.6f}, mean: {conv_kspace.weight.real.mean():.6f}, std: {conv_kspace.weight.real.std():.6f}")
            #     print(f"DEBUG: conv_kspace weight imag - min: {conv_kspace.weight.imag.min():.6f}, max: {conv_kspace.weight.imag.max():.6f}, mean: {conv_kspace.weight.imag.mean():.6f}, std: {conv_kspace.weight.imag.std():.6f}")
            #     print(f"DEBUG: conv_kspace weight abs - min: {conv_kspace.weight.abs().min():.6f}, max: {conv_kspace.weight.abs().max():.6f}, mean: {conv_kspace.weight.abs().mean():.6f}")
            #     # Check if weights look reasonable
            #     if conv_kspace.weight.abs().max() < 1e-6:
            #         print(f"WARNING: conv_kspace weights appear to be zeros or extremely small - may not be loaded correctly!")
            #     print("="*80 + "\n")
            # Freeze slice_label_layers - they are used only for regularization, no gradients needed
            model.slice_label_layers.eval()
            for p in model.slice_label_layers.parameters():
                p.requires_grad = False
            print(f"Frozen {sum(p.numel() for p in model.slice_label_layers.parameters())} parameters in slice_label_layers (no gradients)")
            if config.freeze_backbone:
                print("Freezing backbone and conv_kspace")
                model.backbone.eval()
                for p in model.backbone.parameters():
                    p.requires_grad = False 
                model.conv_kspace.eval()
                for p in model.conv_kspace.parameters():
                    p.requires_grad = False
            else:
                print("Not freezing backbone and conv_kspace")
    else:
        raise NotImplementedError(f"Model type {config.model_type} not implemented")
    return model

