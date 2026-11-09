import numpy as np
from typing import Sequence, Tuple, Optional
import torch
from torch import nn
from classifier.masks.mask_function import MaskFunction
import random


class VariableDensityMask(MaskFunction):
    """
    This class samples acceleration mask randomly but with weights. Weight is formulated in a fashion that higher for
    regions that are closer to center. Similar to Uniform mask, except weight
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
    ):
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
        self, center_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        start, end = self.sampled_indices
        len_sampled_indices = end - start + 1

        # subtract center mask indices
        remaining_indices = set(np.arange(self.target_shape[1]))
        center_mask_indices = set()
        if center_mask is not None:
            center_mask_indices = set(torch.where(center_mask != 0)[0].cpu().numpy())
        remaining_indices -= center_mask_indices

        # keep only indices within start and end
        remaining_indices = {i for i in remaining_indices if i >= start and i <= end}

        # sample random indices from remaining indices
        num_random_indices = round(len_sampled_indices * self.k_fraction) - len(
            center_mask_indices
        )
        num_random_indices = max(0, num_random_indices)

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

if __name__ == "__main__":
    mask = VariableDensityMask(
        (2, 5), k_fraction=0.5, sampled_indices=(0, -1), center_fraction=0.0, batch_size=2
    )
    cm = mask.get_center_mask()
    am = mask.get_acceleration_mask(center_mask=cm)

    print(mask.get_mask().mean())

    x = torch.ones(2, 2, 5)
    print(mask(x, batch_size=2))

