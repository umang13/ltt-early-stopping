import hashlib
import os
from joblib import load
import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.data.cifar10 import cifar10_normalization
from classifier.data.imagenet import imagenet_normalization


def resolve_label_names(config):
    if isinstance(config.label_names, str):
        return config.label_names.split(",")
    if isinstance(config.label_names, list):
        return config.label_names
    raise ValueError(f"label_names must be a string or list, got {type(config.label_names)}")


def resolve_num_classes(config):
    imbalanced_class = getattr(config, "imbalanced_class", None)
    hierarchical_superclass = getattr(config, "hierarchical_superclass", None)
    hierarchical_target_class = getattr(config, "hierarchical_target_class", None)
    is_imbalanced = (imbalanced_class is not None) or (
        hierarchical_superclass is not None and hierarchical_target_class is not None
    )
    if is_imbalanced:
        return 2, True
    return config.num_classes, False


def get_normalization_fn(config):
    if config.dataset == "cifar10":
        return cifar10_normalization()
    if config.dataset in ["imagenet", "imagenet_large", "imagenet_tiny"]:
        return imagenet_normalization()
    return lambda x: x


def init_loss_functions(config, label_names, num_classes, for_validation=False):
    loss_type = getattr(config, "loss_type", "cross_entropy")
    valid = {"cross_entropy", "weighted_cross_entropy", "distillation_kl", "bayes_enforcing"}
    if loss_type not in valid:
        raise NotImplementedError(f"Loss {loss_type} not implemented")

    weights = None
    if loss_type == "weighted_cross_entropy" and not for_validation:
        weights = _load_class_weights(config, label_names, num_classes)

    criterion = {}
    for idx, label_name in enumerate(label_names):
        if loss_type == "weighted_cross_entropy" and weights is not None:
            weight_tensor = weights[idx].float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            criterion[label_name] = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion[label_name] = nn.CrossEntropyLoss()

    return criterion


def _load_class_weights(config, label_names, num_classes):
    # 1) file specified
    weights_file = getattr(config, "loss_fn_weights_filename", None)
    if weights_file and os.path.exists(weights_file):
        return load(weights_file)

    # 2) inline class_weights
    class_weights = getattr(config, "class_weights", None)
    if class_weights is not None:
        return torch.as_tensor(class_weights)

    # 3) fallback to uniform
    return torch.ones(len(label_names), num_classes)


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()


def compute_kl_divergence(full_logits, masked_logits):
    # KL(p_full || p_masked)
    log_probs_full = F.log_softmax(full_logits.detach(), dim=1)
    log_probs_masked = F.log_softmax(masked_logits, dim=1)
    return F.kl_div(log_probs_masked, log_probs_full, reduction="batchmean", log_target=True)


def get_sampling_rate(mask_fn):
    if hasattr(mask_fn, "get_last_sampling_rate"):
        return mask_fn.get_last_sampling_rate()
    if hasattr(mask_fn, "last_sampling_rate"):
        return mask_fn.last_sampling_rate
    return None
