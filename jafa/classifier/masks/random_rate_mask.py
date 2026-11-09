"""
Random Rate Mask Functions

These mask functions randomly select the sampling rate (k_fraction or patch_mask_ratio)
each time they are called, in addition to randomly selecting which elements to mask.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class RandomRateUniformMask:
    """
    K-space mask with randomly selected sampling rate
    
    Each time the mask is called:
    1. Randomly select k_fraction from [min_rate, max_rate]
    2. Apply uniform random k-space masking with that k_fraction
    """
    
    def __init__(
        self,
        target_shape: Tuple[int, int],
        min_rate: float = 0.1,
        max_rate: float = 1.0,
        center_fraction: float = 0.08,
        batch_size: int = 1,
        sampled_indices: Optional[Tuple[int, int]] = None,
        in_channels: int = 1,
        is_complex: bool = True,
        device: torch.device = torch.device("cpu"),
        coil_type: str = "sc",
    ):
        """
        Args:
            target_shape: Shape of k-space (height, width)
            min_rate: Minimum sampling rate
            max_rate: Maximum sampling rate
            center_fraction: Fraction of center k-space to always sample
            batch_size: Batch size
            sampled_indices: Optional fixed indices to sample
            in_channels: Number of input channels
            is_complex: Whether data is complex-valued
            device: Device to use
            coil_type: Coil type ('sc' or 'mc')
        """
        self.target_shape = target_shape
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.center_fraction = center_fraction
        self.batch_size = batch_size
        self.sampled_indices = sampled_indices
        self.in_channels = in_channels
        self.is_complex = is_complex
        self.device = device
        self.coil_type = coil_type
        
        # Calculate center region
        self.center_size = int(center_fraction * target_shape[1])
        self.center_start = (target_shape[1] - self.center_size) // 2
        self.center_end = self.center_start + self.center_size
    
    def __call__(self, kspace: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Apply random rate uniform k-space masking
        
        Args:
            kspace: K-space data [batch, channels, height, width] or [batch, channels, height, width, 2]
            batch_size: Batch size (optional)
        
        Returns:
            Masked k-space data
        """
        if batch_size is None:
            batch_size = kspace.shape[0]
        
        # Randomly select k_fraction for this call
        k_fraction = np.random.uniform(self.min_rate, self.max_rate)
        
        # Calculate number of columns to sample (excluding center)
        total_cols = self.target_shape[1]
        num_cols_to_sample = int(k_fraction * total_cols)
        
        # Always include center columns
        num_cols_to_sample = max(num_cols_to_sample, self.center_size)
        
        # Number of additional columns to sample outside center
        num_outer_cols = num_cols_to_sample - self.center_size
        
        # Create mask
        mask = torch.zeros(batch_size, 1, self.target_shape[0], self.target_shape[1], device=kspace.device)
        
        # Always sample center
        mask[:, :, :, self.center_start:self.center_end] = 1.0
        
        # Randomly sample outer columns
        if num_outer_cols > 0:
            # Get outer column indices (excluding center)
            outer_indices = list(range(0, self.center_start)) + list(range(self.center_end, total_cols))
            
            # Randomly select outer columns for each sample in batch
            for b in range(batch_size):
                selected_outer = np.random.choice(outer_indices, size=num_outer_cols, replace=False)
                mask[b, :, :, selected_outer] = 1.0
        
        # Apply mask
        if self.is_complex and kspace.shape[-1] == 2:
            # Complex data with real/imag in last dimension
            mask = mask.unsqueeze(-1)
        
        masked_kspace = kspace * mask
        
        return masked_kspace


class RandomRatePatchMask:
    """
    Image patch mask with randomly selected masking rate
    
    Each time the mask is called:
    1. Randomly select mask_ratio from [min_rate, max_rate]
    2. Apply random patch masking with that mask_ratio
    """
    
    def __init__(
        self,
        patch_size: int = 8,
        min_rate: float = 0.1,
        max_rate: float = 0.9,
        mask_value: float = 0.0,
    ):
        """
        Args:
            patch_size: Size of each patch
            min_rate: Minimum masking rate (fraction of patches to mask)
            max_rate: Maximum masking rate (fraction of patches to mask)
            mask_value: Value to use for masked patches
        """
        self.patch_size = patch_size
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.mask_value = mask_value
        self.last_sampling_rate = None  # Store the last sampling rate used
    
    def __call__(self, image: torch.Tensor, batch_size: Optional[int] = None, return_mask: bool = False, return_sampling_rate: bool = False):
        """
        Apply random rate patch masking
        
        Args:
            image: Image data [batch, channels, height, width]
            batch_size: Batch size (optional)
            return_mask: If True, return both masked image and binary mask
            return_sampling_rate: If True, return both masked image and sampling rate
        
        Returns:
            Masked image data or (masked_image, binary_mask) or (masked_image, sampling_rate) if return_*=True
        """
        if batch_size is None:
            batch_size = image.shape[0]
        
        # Randomly select mask_ratio for this call
        mask_ratio = np.random.uniform(self.min_rate, self.max_rate)
        
        # Store the sampling rate for potential retrieval
        self.last_sampling_rate = 1.0 - mask_ratio  # Convert mask_ratio to sampling_rate
        
        B, C, H, W = image.shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Number of patches to mask
        num_masked_patches = int(mask_ratio * total_patches)
        
        # Create binary mask
        binary_mask = torch.ones(B, 1, H, W, device=image.device)
        
        # Create masked image (copy original)
        masked_image = image.clone()
        
        # Apply masking for each sample in batch
        for b in range(batch_size):
            # Randomly select patches to mask
            patch_indices = np.random.choice(total_patches, size=num_masked_patches, replace=False)
            
            for patch_idx in patch_indices:
                # Convert flat index to 2D patch coordinates
                patch_i = patch_idx // num_patches_w
                patch_j = patch_idx % num_patches_w
                
                # Calculate pixel coordinates
                start_h = patch_i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = patch_j * self.patch_size
                end_w = start_w + self.patch_size
                
                # Mask the patch
                masked_image[b, :, start_h:end_h, start_w:end_w] = self.mask_value
                # Set binary mask to 0 for masked patches
                binary_mask[b, :, start_h:end_h, start_w:end_w] = 0.0
        
        if return_mask:
            return masked_image, binary_mask
        elif return_sampling_rate:
            return masked_image, self.last_sampling_rate
        else:
            return masked_image
    
    def get_last_sampling_rate(self):
        """Get the last sampling rate used."""
        return self.last_sampling_rate


class BiasedRandomRatePatchMask:
    """
    Image patch mask with sampling rates biased toward lower values.

    The sampling rate is drawn from a mixture distribution that places higher weight
    on values below a specified threshold, encouraging heavily undersampled inputs.
    """

    def __init__(
        self,
        patch_size: int = 8,
        min_sampling_rate: float = 0.05,
        max_sampling_rate: float = 0.9,
        bias_threshold: float = 0.15,
        low_rate_weight: float = 0.7,
        mask_value: float = 0.0,
    ):
        """
        Args:
            patch_size: Size of each square patch.
            min_sampling_rate: Minimum sampling rate allowed (fraction of patches kept).
            max_sampling_rate: Maximum sampling rate allowed.
            bias_threshold: Sampling rates below this value are preferred.
            low_rate_weight: Probability of drawing a sampling rate from the biased low range.
            mask_value: Value used to fill masked patches.
        """
        self.patch_size = patch_size
        self.min_sampling_rate = max(0.0, min_sampling_rate)
        self.max_sampling_rate = min(1.0, max_sampling_rate)
        self.bias_threshold = max(self.min_sampling_rate, min(bias_threshold, self.max_sampling_rate))
        self.low_rate_weight = float(np.clip(low_rate_weight, 0.0, 1.0))
        self.mask_value = mask_value
        self.last_sampling_rate = None

    def _sample_sampling_rate(self) -> float:
        if self.low_rate_weight <= 0.0 or self.bias_threshold <= self.min_sampling_rate:
            return float(np.random.uniform(self.min_sampling_rate, self.max_sampling_rate))

        if np.random.rand() < self.low_rate_weight:
            low_max = min(self.bias_threshold, self.max_sampling_rate)
            return float(np.random.uniform(self.min_sampling_rate, low_max))
        else:
            high_min = max(self.bias_threshold, self.min_sampling_rate)
            return float(np.random.uniform(high_min, self.max_sampling_rate))

    def __call__(
        self,
        image: torch.Tensor,
        batch_size: Optional[int] = None,
        return_mask: bool = False,
        return_sampling_rate: bool = False,
    ):
        if batch_size is None:
            batch_size = image.shape[0]

        sampling_rate = self._sample_sampling_rate()
        mask_ratio = 1.0 - sampling_rate
        mask_ratio = float(np.clip(mask_ratio, 0.0, 1.0))
        self.last_sampling_rate = sampling_rate

        B, C, H, W = image.shape

        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        total_patches = num_patches_h * num_patches_w

        num_masked_patches = int(mask_ratio * total_patches)

        binary_mask = torch.ones(B, 1, H, W, device=image.device)
        masked_image = image.clone()

        for b in range(batch_size):
            if num_masked_patches <= 0:
                continue
            patch_indices = np.random.choice(total_patches, size=num_masked_patches, replace=False)
            for patch_idx in patch_indices:
                patch_i = patch_idx // num_patches_w
                patch_j = patch_idx % num_patches_w

                start_h = patch_i * self.patch_size
                end_h = start_h + self.patch_size
                start_w = patch_j * self.patch_size
                end_w = start_w + self.patch_size

                masked_image[b, :, start_h:end_h, start_w:end_w] = self.mask_value
                binary_mask[b, :, start_h:end_h, start_w:end_w] = 0.0

        if return_mask:
            return masked_image, binary_mask
        elif return_sampling_rate:
            return masked_image, self.last_sampling_rate
        else:
            return masked_image

    def get_last_sampling_rate(self):
        return self.last_sampling_rate

