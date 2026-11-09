import numpy as np
from typing import Sequence, Tuple, Optional
import torch
from torch import nn
from .mask_function import MaskFunction
import random


class EquiSpacedMask(MaskFunction):
    """
    This class sample acceleration mask with equal space in between. This class is not trainable.
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

        center_mask_indices = list(center_mask_indices)
        if len(center_mask_indices) == 0:
            left_index = -1
            right_index = -1
            left_remaining_indices = remaining_indices
            right_remaining_indices = remaining_indices
        else:
            left_index = center_mask_indices[0]
            right_index = center_mask_indices[-1]

            left_remaining_indices = {
                i for i in remaining_indices if i >= start and i <= left_index
            }
            right_remaining_indices = {
                i for i in remaining_indices if i >= right_index and i <= end
            }

        # sample random indices from remaining indices
        num_random_indices = round(len_sampled_indices * self.k_fraction) - len(
            center_mask_indices
        )
        num_random_indices = max(0, num_random_indices)
        if num_random_indices == 0:
            # create mask
            _, num_cols = self.target_shape
            mask = torch.zeros(num_cols).float()
            return mask.to(self.device)


        acc_factor = len_sampled_indices // num_random_indices
        # print(num_random_indices)

        remaining_indices = np.asarray(list(remaining_indices))

        left_remaining_indices = np.asarray(list(left_remaining_indices))
        right_remaining_indices = np.sort(np.asarray(list(right_remaining_indices)))

        left_remaining_indices = np.arange(
            left_remaining_indices[0], left_remaining_indices[-1] + 1, step=acc_factor
        )
        right_remaining_indices = np.arange(
            right_remaining_indices[0], right_remaining_indices[-1] + 1, step=acc_factor
        )

        # create mask
        _, num_cols = self.target_shape
        mask = torch.zeros(num_cols).float()

        mask[left_remaining_indices] = 1
        mask[right_remaining_indices] = 1

        return mask.to(self.device)


if __name__ == "__main__":
    mask = EquiSpacedMask(
        (0, 100), k_fraction=0.08, sampled_indices=(0, -1), center_fraction=0
    )
    cm = mask.get_center_mask()
    am = mask.get_acceleration_mask(center_mask=cm)

    print(mask.get_mask().mean())

    x = torch.randn(1, 1, 100)
    print(mask(x))
