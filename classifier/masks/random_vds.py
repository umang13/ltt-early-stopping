import numpy as np
from typing import Sequence, Tuple, Optional
import torch
from torch import nn
from classifier.masks.mask_function import MaskFunction
import random


class RandomVariableDensityMask(MaskFunction):
    """
    This class samples acceleration mask randomly but with weights. Weight is formulated in a fashion that higher for
    regions that are closer to center. Similar to Uniform mask, except weight.
    
    Unlike VariableDensityMask, this class randomly samples k_fraction (and center_fraction) 
    uniformly from a specified range for each mask generation.
    """
    def __init__(
        self,
        target_shape: Tuple[int, int],
        batch_size: int,
        k_fraction: float,
        sampled_indices: Tuple[int, int] = (0, -1),
        center_fraction: float = 0.0,
        in_channels: int = 1,
        is_complex: bool = True,
        device: torch.device = torch.device("cpu"),
        min_k_fraction: Optional[float] = None,
        max_k_fraction: Optional[float] = None,
    ):
        """
        Args:
            min_k_fraction: Minimum k_fraction to sample from. If None, uses center_fraction as minimum.
            max_k_fraction: Maximum k_fraction to sample from. If None, uses k_fraction as maximum.
        """
        super().__init__(
            target_shape=target_shape,
            batch_size=batch_size,
            k_fraction=k_fraction,
            sampled_indices=sampled_indices,
            center_fraction=center_fraction,
            in_channels=in_channels,
            is_complex=is_complex,
            device=device,
        )
        # Store original k_fraction as max, and center_fraction as min (or provided min)
        self.max_k_fraction = max_k_fraction if max_k_fraction is not None else 1.0
        self.min_k_fraction = min_k_fraction if min_k_fraction is not None else 0.05
        
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

    def sample_mask(self, shape: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override the base class sample_mask to randomly sample k_fraction before generating masks.
        """
        # Randomly sample k_fraction from the range
        # self.k_fraction = np.random.uniform(self.min_k_fraction, self.max_k_fraction)
        # Center fraction is always k_fraction / 2
        # self.center_fraction = self.k_fraction / 2
        # print(f"k fraction : {self.k_fraction}, center fraction : {self.center_fraction}")
        if shape:
            center_mask = self.get_center_mask(shape)
            accel_mask = self.get_acceleration_mask(center_mask)
        else:
            center_mask = self.get_center_mask()
            accel_mask = self.get_acceleration_mask(center_mask)
        return center_mask, accel_mask

    def get_acceleration_mask(
        self, center_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        start, end = self.sampled_indices
        len_sampled_indices = end - start + 1

        # subtract center mask indices
        remaining_indices = set(np.arange(self.target_shape[1]))
        center_mask_indices = set()
        if center_mask is not None:
            center_mask_indices = set(torch.where(center_mask != 0)[0].cpu().numpy())
        # print(len(remaining_indices), len(center_mask_indices))
        remaining_indices -= center_mask_indices

        # keep only indices within start and end
        remaining_indices = {i for i in remaining_indices if i >= start and i <= end}

        # sample random indices from remaining indices
        num_random_indices = round(len_sampled_indices * self.k_fraction) - len(center_mask_indices)
        num_random_indices = max(0, num_random_indices)
        # print("num_random_indices remaining after subtracting center mask indices", num_random_indices)
        # create mask
        _, num_cols = self.target_shape
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

    def forward(self, x: torch.Tensor, shape: Optional[Sequence[int]] = None, num_slices: Optional[int] = None, multi_coil: bool = False, return_mask: bool = False) -> torch.Tensor:
        """
        Override forward to generate a different mask for each sample in the batch.
        Each sample gets a randomly sampled k_fraction from the range.
        """
        batch_size = x.shape[0]
        
        if num_slices:
            # For volumetric case, generate masks per sample
            masks = []
            for i in range(batch_size):
                # Randomly sample k_fraction for this sample
                self.k_fraction = np.random.uniform(self.min_k_fraction, self.max_k_fraction)
                # Center fraction is always k_fraction / 2
                self.center_fraction = self.k_fraction / 2
                # print("k_fraction in random_vds", self.k_fraction)
                # Generate mask for num_slices
                sample_masks = [self.get_mask(shape) for _ in range(num_slices)]
                sample_mask = torch.stack(sample_masks, dim=0)
                sample_mask = sample_mask.unsqueeze(1)
                if multi_coil:
                    sample_mask = sample_mask.unsqueeze(1)
                masks.append(sample_mask)
            
            mask = torch.stack(masks, dim=0)
            # mask = mask.unsqueeze(-2)
        else:
            # For non-volumetric case, generate one mask per sample
            masks = []
            for i in range(batch_size):
                # Randomly sample k_fraction for this sample
                self.k_fraction = np.random.uniform(self.min_k_fraction, self.max_k_fraction)
                # print("batch index", i, "k_fraction", self.k_fraction)
                # Center fraction is always k_fraction / 2
                self.center_fraction = self.k_fraction / 2
                
                # Generate mask for this sample
                sample_mask = self.get_mask(shape)
                masks.append(sample_mask)
            
            mask = torch.stack(masks, dim=0)

            # Reshape mask to broadcast correctly with x
            # mask shape is [batch_size, num_cols] or [batch_size, 1, num_cols]
            # Need to add dimensions to match x's shape for proper broadcasting
            # The mask should be applied along the last dimension (k-space columns)
            if len(mask.shape) == 2:  # [batch_size, num_cols]
                # Add singleton dimensions to match x shape (except batch and last dim)
                for _ in range(len(x.shape) - len(mask.shape)):
                    mask = mask.unsqueeze(1)
        # print("mask.shape and kspace.shape in forward", mask.shape, x.shape)
        mask = mask.to(x.device)
        # print("Percentage of ones in mask", (mask == 1).sum()/mask.numel())
        if return_mask:
            return x * mask, mask
        else:
            return x * mask

if __name__ == "__main__":
    mask = RandomVariableDensityMask(
        (2, 5), k_fraction=0.5, sampled_indices=(0, -1), center_fraction=0.0, batch_size=2,
        min_k_fraction=0.2, max_k_fraction=0.8
    )
    cm = mask.get_center_mask()
    am = mask.get_acceleration_mask(center_mask=cm)

    print(mask.get_mask().mean())

    x = torch.ones(2, 2, 5)
    print(mask(x, batch_size=2))

