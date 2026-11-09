import numpy as np
import torch
from typing import Sequence, Tuple, Optional
from torch import nn
import torch.nn.functional as F

from .mask_function import MaskFunction
from .uniform import UniformRandomMask
from classifier.classification_modules.arms_module import ARMSClassifier


class GreedyMaskSelection:
    def __init__(self, config, init_klines, ckpt_name) -> None:
        self.config = config
        self.init_klines = init_klines

        # load model
        self.model = ARMSClassifier(config=self.config)
        self.model = self.model.load_from_checkpoint(ckpt_name)

        self.mask_sequence = {}

    def init_mask(self):
        k_fraction = self.init_klines / self.config.target_shape[0]
        mask_fn = UniformRandomMask(
            target_shape=self.config.target_shape,
            k_fraction=k_fraction,
            center_fraction=k_fraction,
            sampled_indices=self.config.sampled_indices,
            device=self.device,
        )

        if self.config.trajectory_type == "cartesian":
            self.mask_sequence[self.init_klines] = mask_fn.get_mask()
        else:
            raise NotImplementedError

    def generate_mask(self, mask_selection_idx):
        raise NotImplementedError

    def mask_selection(self, datamodule, ckpt_name):
        raise NotImplementedError
