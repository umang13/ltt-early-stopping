from typing import Dict

import torchmetrics.classification
from sklearn import metrics
import torch
from torchmetrics import functional
from torchmetrics.classification import BinaryAUROC
import numpy as np


def compute_auc(preds, labels, num_classes):
    metric = BinaryAUROC()
    return metric(
        preds=preds,
        target=labels,
    )


def compute_auc_multiclass(preds, labels, num_classes):
    auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    # need to use cpu. For Cuda: RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
    return auroc(preds=preds.detach().cpu(), target=labels.detach().cpu())


def compute_accuracy(preds, labels, num_classes):
    accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
    return accuracy(preds=preds.detach().cpu(), target=labels.detach().cpu())


def get_operating_point(preds, labels, operating_point=None, threshold=0.1):
    preds = preds.cpu()
    preds_positive = preds[:, 1].numpy()

    # Ensure labels are 1D before converting to numpy
    if len(labels.shape) > 1:
        labels = labels.squeeze(1)
    labels = labels.cpu().numpy().astype(int)
    # Flatten in case it's still not 1D
    labels = labels.flatten()

    if operating_point is None:
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds_positive)

        operating_point = thresholds[fpr > 0.25][0]

        try:
            fnr = 1 - tpr
            operating_point = thresholds[fnr < threshold][0]
        except IndexError:
            operating_point = thresholds[fpr > 0.25][0]

    test_predictions = (preds_positive > operating_point).astype(int)

    sensitivity = metrics.recall_score(y_true=labels, y_pred=test_predictions)
    specificity = metrics.recall_score(
        y_true=np.abs(1 - labels), y_pred=np.abs(1 - test_predictions)
    )
    balanced_acc = metrics.balanced_accuracy_score(
        y_true=labels, y_pred=test_predictions
    )

    precision = metrics.precision_score(y_true=labels, y_pred=test_predictions)
    npv = metrics.precision_score(y_true=1.0 - labels, y_pred=1.0 - test_predictions)

    return balanced_acc, specificity, sensitivity, precision, operating_point, npv


def evaluate_classifier(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    operating_point: float = None,
) -> Dict:

    # Ensure labels are 1D for all computations
    if len(labels.shape) > 1:
        labels_1d = labels.squeeze(1)
    else:
        labels_1d = labels
    # print("##############>>>>>>", preds.shape, labels_1d.shape, labels.shape)
    # Compute accuracy
    accuracy = compute_accuracy(preds=preds, labels=labels_1d, num_classes=num_classes)
    
    try:
        if num_classes == 2:
            assert preds.shape[1] == 2, "preds should be of shape (batch_size, 2)"
            # Ensure labels_1d is used for AUC computation
            auc = compute_auc(preds=preds[:, 1], labels=labels_1d, num_classes=num_classes).item()
            (
                balanced_acc,
                specificity,
                sensitivity,
                precision,
                operating_point,
                npv,
            ) = get_operating_point(preds, labels_1d, operating_point=operating_point)
        else:
            auc = compute_auc_multiclass(preds=preds, labels=labels, num_classes=num_classes).item()
            balanced_acc = np.nan
            specificity = np.nan
            sensitivity = np.nan
            precision = np.nan
            operating_point = np.nan
            npv = np.nan

    except ValueError:
        auc = np.nan
        balanced_acc = np.nan
        specificity = np.nan
        sensitivity = np.nan
        precision = np.nan
        operating_point = np.nan
        npv = np.nan

    ret = dict(
        auc=auc,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        balanced_accuracy=balanced_acc,
        accuracy=accuracy,
        operating_point=operating_point,
        npv=npv,
    )

    return ret


def classifier_metrics(
    val_preds: torch.Tensor,
    val_labels: torch.Tensor,
    test_preds: torch.Tensor,
    test_labels: torch.Tensor,
    threshold: float,
):
    _, _, _, _, operating_point, _ = get_operating_point(
        preds=val_preds, labels=val_labels, operating_point=None, threshold=threshold
    )

    return evaluate_classifier(
        preds=test_preds, labels=test_labels, operating_point=operating_point
    )
