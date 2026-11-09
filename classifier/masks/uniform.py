import numpy as np
from typing import Sequence, Tuple, Optional
import torch
from torch import nn
from .mask_function import MaskFunction
import random


class UniformRandomMask(MaskFunction):
    """
    This class samples acceleration mask with uniform distribution. This class is not trainable
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

    def get_acceleration_mask(
        self, center_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param center_mask: (1D tensor) already sampled center mask
        :return: (1D tensor) acceleration mask
        """
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
            random_indices = torch.Tensor(
                random.sample(list(remaining_indices), num_random_indices)
            ).long()
            mask[random_indices] = 1

        return mask.to(self.device)



if __name__ == "__main__":
    mask = UniformRandomMask(
        (5, 1000), k_fraction=0.5, sampled_indices=(0, -1), center_fraction=0.15
    )
    cm = mask.get_center_mask()
    am = mask.get_acceleration_mask(center_mask=cm)

    print(mask.get_mask().mean())

    x = torch.randn(2, 5, 1000)
    print(mask(x))
