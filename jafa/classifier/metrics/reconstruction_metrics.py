import torch
from typing import Optional
from torch import nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = 0.0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]



def recon_metrics(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    return dict of SSIM, PSNR, MSE, NMSE, between recon and target images
    """
    target = target.cpu().numpy()
    recon = recon.cpu().numpy()
    n_images = recon.shape[0]

    metrics = dict(ssim=0, psnr=0, mse=0, nmse=0)
    for i in range(n_images):
        metrics["ssim"] += ssim(gt=target[i], pred=recon[i]) / n_images
        metrics["psnr"] += psnr(gt=target[i], pred=recon[i]) / n_images
        metrics["mse"] += mse(gt=target[i], pred=recon[i]) / n_images
        metrics["nmse"] += nmse(gt=target[i], pred=recon[i]) / n_images

    return metrics


if __name__ == "__main__":
    print(recon_metrics(torch.ones(2, 1, 10, 10), torch.ones(2, 1, 10, 10)))