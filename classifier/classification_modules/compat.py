import argparse
from typing import Tuple

import torch

from classifier.classification_modules.modules import ARMSModule, BayesDistillModule


class _CompatBase:
    def forward(self, batch, *args, **kwargs):
        return super().forward(batch)


class CIFAR10Classifier(_CompatBase, ARMSModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


class ImageNetClassifier(_CompatBase, ARMSModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


class BayesCIFAR10Classifier(_CompatBase, BayesDistillModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


class BayesImageNetClassifier(_CompatBase, BayesDistillModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


# Aliases for k-space scripts expecting these names
class ARMSClassifier(_CompatBase, ARMSModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


class BayesDistillArmsClassifier(_CompatBase, BayesDistillModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


class BayesDistillClassifier(_CompatBase, BayesDistillModule):
    def __init__(self, config: argparse.Namespace):
        super().__init__(config)


def load_state_dict_with_classifier_fallback(
    model,
    ckpt_path: str,
    map_location="cpu",
    strict: bool = False,
    weights_only: bool = False,
) -> Tuple[list, list]:
    """
    Load a checkpoint into a model, remapping 'classifier.*' keys to 'model.*' if needed.

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=weights_only)
    state_dict = ckpt.get("state_dict", ckpt)
    has_classifier_keys = any(k.startswith("classifier.") for k in state_dict.keys())
    has_model_keys = any(k.startswith("model.") for k in state_dict.keys())
    if has_classifier_keys and not has_model_keys and hasattr(model, "model"):
        remapped = {
            ("model." + k[len("classifier."):]) if k.startswith("classifier.") else k: v
            for k, v in state_dict.items()
        }
        return model.load_state_dict(remapped, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


def infer_pool_type_from_state_dict(state_dict, default_pool_type=None):
    """
    Infer pool_type based on checkpoint keys to avoid importance network mismatches.
    """
    keys = list(state_dict.keys())
    if any("class_importance_networks" in k for k in keys):
        return "class_specific_importance"
    if any("importance_network" in k for k in keys):
        return "learned_weight"
    return default_pool_type
