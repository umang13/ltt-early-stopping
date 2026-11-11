import argparse
import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import joblib

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, sampler
from torchvision import transforms
from tqdm.auto import tqdm


class MRVolumeDataset(Dataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
        dev_mode: bool = False,
        transforms: transforms.Compose = None,
    ):
        super().__init__()
        # read csv file with filenames
        self.config = config
        self.config.split_csv_file = Path(self.config.split_csv_file)
        assert self.config.split_csv_file.is_file()

        self.metadata = pd.read_csv(self.config.split_csv_file)

        # only include slice zero in the metadata file
        self.vol_metadata = self.metadata[self.metadata.slice_id == 0]

        if dev_mode:
            self.vol_metadata = self.vol_metadata.iloc[:50]

        assert "data_split" in self.metadata.columns
        assert "location" in self.metadata.columns

        vol_metadata_grouped = self.vol_metadata.groupby("data_split")
        self.metadata_by_mode = {
            e: vol_metadata_grouped.get_group(e) for e in vol_metadata_grouped.groups
        }

        self.transforms = transforms

        self.mode = mode
        assert mode in self.metadata_by_mode

    def __len__(self):
        assert self.mode in self.metadata_by_mode
        return self.metadata_by_mode[self.mode].shape[0]

    def apply_transforms(self, image, kspace):
        # print(self.transforms)
        # print("Before transforms : ", kspace.shape)
        if self.transforms is not None:
            # print("Enters here")
            if image is not np.nan and kspace is not np.nan:
                # print("Should not enter here")
                # TODO need to zero-pad the image tensor to do a stacked transform

                # stacked_tensor = np.stack([image, kspace.real, kspace.imag])
                # stacked_tensor = self.transforms(stacked_tensor)

                # image = stacked_tensor[0]

                # kspace_real = stacked_tensor[1]
                # kspace_imag = stacked_tensor[2]
                # kspace = kspace_real + 1j * kspace_imag

                image = self.transforms(image)

                stacked_tensor = np.stack([kspace.real, kspace.imag], axis=2)

                stacked_tensor = self.transforms(stacked_tensor)

                kspace = stacked_tensor[0] + 1j * stacked_tensor[1]

            elif image is not None and kspace is np.nan:
                image = self.transforms(image)

            elif image is None and kspace is not np.nan:
                print("Applying transforms to kspace : {self.transforms}")
                stacked_tensor = np.stack([kspace.real, kspace.imag], axis=2)
                stacked_tensor = self.transforms(stacked_tensor)

                kspace = stacked_tensor[0] + 1j * stacked_tensor[1]
        # print("After transforms : ", kspace.shape)
        return image, kspace

    def __getitem__(self, index):
        raise NotImplementedError
