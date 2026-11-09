from typing import Sequence, Optional, Tuple
from torch import nn
import torch
import numpy as np


class MaskFunction(nn.Module):
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
    ):
        """
        The `MaskFunction` class is a base class for creating masks used in k-space sampling for MRI data.
        It is designed to be extended by specific mask implementations that define how the k-space is sampled.

        The constructor initializes several parameters:
        
        - `target_shape`: Specifies the shape of the k-space data that the mask will be applied to.
        - `k_fraction`: Determines the fraction of k-space that will be sampled. This is a key parameter for controlling the amount of data retained.
        - `sampled_indices`: A tuple indicating the range of indices in k-space that are eligible for sampling. By default, it covers the entire range.
        - `center_fraction`: Specifies the fraction of low-frequency components in k-space that should always be sampled. This is important for preserving image quality.
        - `is_complex`: A boolean flag indicating whether the k-space data is complex-valued, which is typical in MRI.
        - `device`: The computational device (CPU or GPU) where the mask will be created and used.

        The class also adjusts the `sampled_indices` to ensure they are within the bounds of the `target_shape`.

        This class provides methods for reshaping masks, sampling masks, and defining the acceleration mask, but these methods are intended to be overridden by subclasses.
        """
        super().__init__()
        self.target_shape = target_shape
        self.center_fraction = center_fraction
        self.k_fraction = k_fraction
        self.is_complex = is_complex
        self.device = device
        self.batch_size = batch_size
        self.in_channels = in_channels

        sampled_indices = (
            (sampled_indices[0], target_shape[1] + sampled_indices[1])
            if sampled_indices[1] < 0
            else sampled_indices
        )
        self.sampled_indices = sampled_indices


    def reshape_mask(self, mask: np.ndarray, shape) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def sample_mask(self,shape: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if shape:
            num_cols = shape[-2]
            center_fraction, acceleration = self.choose_acceleration()
            # num_low_frequencies = round(num_cols * center_fraction)
            center_mask = self.reshape_mask(self.get_center_mask(shape), shape)
            acceleration_mask = self.reshape_mask(self.get_acceleration_mask(center_mask, num_cols, acceleration), shape)
        else:
            center_mask = self.get_center_mask()
            acceleration_mask = self.get_acceleration_mask(center_mask)
        return center_mask, acceleration_mask

    def choose_acceleration(self): #Equispaced Specific. Needs to be updated to work properly
        raise NotImplementedError

    def get_acceleration_mask(self, center_mask: Optional[torch.Tensor] = None, num_cols: Optional[int] = None, acceleration: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    def get_mask(self, shape: Optional[Sequence[int]] = None) -> torch.Tensor:
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
 
    def forward(self, x: torch.Tensor,shape: Optional[Sequence[int]]=None, num_slices: Optional[int]=None, multi_coil: bool=False, return_mask: bool=False) -> torch.Tensor:
        if num_slices:
            mask = [self.get_mask(shape) for _ in range(num_slices)]
            mask = torch.stack(mask, dim=0)
            mask = mask.unsqueeze(1)
            if multi_coil:
                mask = mask.unsqueeze(1)
                assert mask.shape[1] == 1
                assert mask.shape[2] == 1
        else:
            # breakpoint()
            mask = self.get_mask(shape)
        
        # breakpoint()
        mask = mask.to(x.device)
        if return_mask:
            return x * mask, mask
        else:
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
            num_cols = shape[-2]
            num_low_freqs = round(num_cols * self.center_fraction)
            mask = np.zeros(num_cols, dtype=np.float32)
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


if __name__ == "__main__":
    mask = MaskFunction((5, 7), 0.5, center_fraction=1/3, sampled_indices=(4, 6))
    x = torch.ones(5 * 7).reshape(5, 7)

    cm = mask.get_center_mask()
    print(cm)
