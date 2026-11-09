import numpy as np
from typing import Sequence, Tuple, Optional
import torch
from torch import nn
from classifier.masks.mask_function import MaskFunction
import random


class RandomVolumetricVariableDensityMask(MaskFunction):
    """
    Variable Density Sampling (VDS) mask with random k_fraction selection.
    
    Unlike VolumetricVariableDensityMask, this class randomly samples k_fraction 
    (and center_fraction) uniformly from a specified range for each mask generation.
    """
    def __init__(
        self,
        target_shape: Tuple[int, int],
        k_fraction: float,
        batch_size: int,
        in_channels: int = 1,
        sampled_indices: Tuple[int, int] = (0, -1),
        center_fraction: float = 0.0,
        is_complex: bool = True,
        device: torch.device = torch.device("cpu"),
        min_k_fraction: Optional[float] = 0.05,
        max_k_fraction: Optional[float] = None,
    ):
        """
        Variable Density Sampling (VDS) mask with random k_fraction.

        target_shape: Tuple[int, int] = is the shape of the k-space.
        k_fraction: float = defines the maximum percentage of line to sample other than the center fraction.
        sampled_indices: Tuple[int, int] = are the indices of the k-space that are to be taken into consideration for the sampling.
        center_fraction: float = is the minimum fraction of the k-space that is sampled in the center.
        is_complex: bool = is a boolean that indicates if the k-space is complex.
        device: torch.device = is the device on which the mask is created.
        min_k_fraction: Minimum k_fraction to sample from. If None, uses center_fraction as minimum.
        max_k_fraction: Maximum k_fraction to sample from. If None, uses k_fraction as maximum.
        """    
        super().__init__(
            target_shape=target_shape,
            k_fraction=k_fraction,
            batch_size=batch_size,
            in_channels=in_channels,
            sampled_indices=sampled_indices,
            center_fraction=center_fraction,
            is_complex=is_complex,
            device=device,
        )
        # Store original k_fraction as max, and center_fraction as min (or provided min)
        self.max_k_fraction = max_k_fraction if max_k_fraction is not None else k_fraction
        self.min_k_fraction = min_k_fraction if min_k_fraction is not None else 0.05
        self.orig_center_fraction = center_fraction
        
        # Ensure min <= max
        if self.min_k_fraction > self.max_k_fraction:
            raise ValueError(f"min_k_fraction ({self.min_k_fraction}) must be <= max_k_fraction ({self.max_k_fraction})")

    def get_prior(self, remaining_indices: Sequence[int]) -> np.ndarray:

        n_cols = len(remaining_indices)

        if n_cols % 2 == 0:
            dist = np.arange(1, n_cols // 2 + 1)
            dist = np.r_[dist, dist[::-1]]
        else:
            dist = np.arange(1, n_cols // 2 + 2)
            dist = np.r_[dist, dist[::-1][:-1]]
        return dist / dist.sum()

    def get_acceleration_mask(
        self, center_mask: Optional[torch.Tensor] = None, shape: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """
        This function generates the acceleration mask for the Variable Density Sampling (VDS) mask.
        The acceleration mask is used to randomly sample k-space indices based on a variable density
        distribution, which is determined by a randomly sampled k_fraction parameter.

        The function first calculates the remaining indices that are not part of the center mask.
        It then samples a specified number of random indices from these remaining indices, based on
        a prior distribution. The prior distribution is calculated using the get_prior method, which
        creates a symmetric distribution around the center of the k-space.

        Args:
            center_mask: The center mask for the VDS mask. This is a tensor containing the indices
                         of the low-frequency components of k-space that are always sampled.
            shape: Optional shape parameter for determining number of columns.

        Returns:
            The acceleration mask for the VDS mask. This is a tensor containing the indices of the
            high-frequency components of k-space that are sampled according to the variable density
            distribution.
        """
        start, end = self.sampled_indices
        len_sampled_indices = end - start + 1

        # Determine the number of columns based on shape or target_shape
        if shape:
            num_cols = shape[-1]
        else:
            _, num_cols = self.target_shape
            
        # subtract center mask indices
        remaining_indices = set(np.arange(num_cols))
        center_mask_indices = set()
        if center_mask is not None:
            center_mask_indices = set(torch.where(center_mask != 0)[0].cpu().numpy())
        remaining_indices -= center_mask_indices

        # keep only indices within start and end
        remaining_indices = {i for i in remaining_indices if i >= start and i <= end}

        # sample random indices from remaining indices
        # maintains k_fraction of the total number of indices
        num_random_indices = round(len_sampled_indices * self.k_fraction) - len(center_mask_indices)
        num_random_indices = max(0, num_random_indices)
        
        # create mask
        mask = torch.zeros(num_cols).float()

        # if there are some mask indices to fill in, fill them in randomly

        if num_random_indices > 0:
            sorted_remaining_indices = sorted(list(remaining_indices))
            sampling_dist = self.get_prior(sorted_remaining_indices)

            random_indices = torch.Tensor(
                np.random.choice(
                    sorted_remaining_indices,
                    num_random_indices,
                    replace=False,
                    p=sampling_dist,
                )
            ).long()

            mask[random_indices] = 1

        return mask.to(self.device)

    def sample_mask(self, shape: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override the base class sample_mask to randomly sample k_fraction before generating masks.
        """
        # Randomly sample k_fraction from the range
        self.k_fraction = np.random.uniform(self.min_k_fraction, self.max_k_fraction)
        # Adjust center_fraction proportionally (maintain the ratio if orig_center_fraction was set)
        if self.orig_center_fraction > 0:
            # Maintain the ratio between center_fraction and k_fraction
            ratio = self.orig_center_fraction / (self.max_k_fraction if self.max_k_fraction > 0 else 1.0)
            self.center_fraction = self.k_fraction * ratio
        else:
            # If no center_fraction was originally set, use k_fraction / 2 as default
            self.center_fraction = self.k_fraction / 2
        
        if shape:
            center_mask = self.get_center_mask(shape)
            accel_mask = self.get_acceleration_mask(center_mask, shape)
        else:
            center_mask = self.get_center_mask()
            accel_mask = self.get_acceleration_mask(center_mask)
        return center_mask, accel_mask

    def get_mask(self, shape: Optional[Sequence[int]] = None, k_fraction: Optional[float] = None) -> torch.Tensor:

        # If k_fraction is passed in, use it (for volumetric case with per-slice k_fraction)
        # Otherwise, randomly sample k_fraction from the range
        if k_fraction is not None:
            self.k_fraction = k_fraction
            self.center_fraction = k_fraction / 2
            # When k_fraction is explicitly provided, don't randomize in sample_mask
            # So we'll call get_center_mask and get_acceleration_mask directly
            if shape and len(shape) < 3:
                raise ValueError("Shape should have 3 or more dimensions")
            
            if shape:
                center_mask = self.get_center_mask(shape)
                accel_mask = self.get_acceleration_mask(center_mask, shape)
            else:
                center_mask = self.get_center_mask()
                accel_mask = self.get_acceleration_mask(center_mask)
        else:
            # Randomly sample k_fraction from the range via sample_mask
            if shape and len(shape) < 3:
                raise ValueError("Shape should have 3 or more dimensions")
            
            if shape:
                center_mask, accel_mask = self.sample_mask(shape)
            else:
                center_mask, accel_mask = self.sample_mask()
        
        # combine masks together
        mask = torch.max(center_mask, accel_mask).to(self.device)

        if not self.is_complex:
            return mask.unsqueeze(1)

        return mask        


    def forward(self, x: torch.Tensor,shape: Optional[Sequence[int]]=None, num_slices: Optional[int]=None) -> torch.Tensor:
        if num_slices:
            # For volumetric case, randomly sample k_fraction for each slice
            # Sample from the range for each slice independently
            # range_k_fraction = np.random.uniform(
            #     self.min_k_fraction, 
            #     self.max_k_fraction, 
            #     size=num_slices
            # )
            # range_k_fraction = torch.from_numpy(range_k_fraction).float()
            
            # assert len(range_k_fraction) == num_slices
            # mask = [self.get_mask(shape, k_fraction=k_fraction.item()) for k_fraction in range_k_fraction]
            # mask = torch.stack(mask, dim=0)
            # mask = mask.unsqueeze(1)
            self.max_k_fraction = np.random.uniform(self.min_k_fraction, 1.0)
            # print("Max k fraction", self.max_k_fraction)
            center_idx = num_slices // 2
            center_idx = (num_slices - 1) / 2  # float, supports both even & odd
            indices = torch.arange(num_slices)
            # Symmetric triangle function for a single peak
            range_k_fraction = self.max_k_fraction - torch.abs(indices - center_idx) * ((self.max_k_fraction - self.min_k_fraction) / center_idx)            
            
            # Print effective k_fraction for the whole volume (mean across all slices)
            effective_k_fraction = range_k_fraction.mean().item()
            # print(f"Effective k_fraction for whole volume (mean across {num_slices} slices): {effective_k_fraction:.4f}")

            # print(f"{range_k_fraction=}")
            assert len(range_k_fraction) == num_slices
            mask = [self.get_mask(shape, k_fraction=k_fraction.item()) for k_fraction in range_k_fraction]
            mask = torch.stack(mask, dim=0)
            mask = mask.unsqueeze(1)
        else:
            mask = self.get_mask(shape)

        mask = mask.to(x.device)
        # print("x.shape", x.shape, "mask.shape", mask.shape)
        return x * mask 


    def get_center_mask(self, shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        """
        Build center mask based on number of low frequencies.
        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.
        Returns:
            A mask for the low spatial frequencies of k-space.
        """
        if shape: #VarNet Masking Code
            num_cols = shape[-1]
            num_low_freqs = round(num_cols * self.center_fraction)
            mask = torch.zeros(num_cols, dtype=torch.float32).to(self.device)
            pad = (num_cols - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = 1
            assert mask.sum() == num_low_freqs

        else: #Old Code
            start, end = self.sampled_indices

            len_sampled_indices = end - start + 1

            num_low_frequencies = round(len_sampled_indices * self.center_fraction)

            _, num_cols = self.target_shape
            mask = torch.zeros(num_cols).float().to(self.device)
            if num_low_frequencies == 0:
                return mask

            pad = (len_sampled_indices - num_low_frequencies + 1) // 2
            mask[start + pad : start + pad + num_low_frequencies] = 1
            assert mask.sum() == num_low_frequencies
        return mask               


# if __name__ == "__main__":
#     mask = RandomVolumetricVariableDensityMask(
#         (5, 1000), k_fraction=0.5, sampled_indices=(0, -1), center_fraction=0.0,
#         min_k_fraction=0.2, max_k_fraction=0.8
#     )
#     cm = mask.get_center_mask()
#     am = mask.get_acceleration_mask(center_mask=cm)

#     print(mask.get_mask().mean())

#     x = torch.randn(2, 5, 1000)
#     print(mask(x))

