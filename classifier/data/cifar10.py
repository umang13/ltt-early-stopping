import argparse
import math
import os
from collections import namedtuple
from pathlib import Path

from classifier.utils.transforms import ifftc, fftc
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
# CIFAR-10 normalization values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

def cifar10_normalization():
    """CIFAR-10 normalization transform"""
    return transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

CIFAR_SPLIT = torch.arange(50000)
CIFAR_TRAIN_IDXS, CIFAR_VAL_IDXS = torch.utils.data.random_split(
                CIFAR_SPLIT, [45000, 5000]
            )
CIFAR_TRAIN_IDXS, CIFAR_VAL_IDXS = torch.tensor(CIFAR_TRAIN_IDXS), torch.tensor(CIFAR_VAL_IDXS)
class kspaceCIFAR10(Dataset):
    def __init__(self, config: argparse.ArgumentParser, mode: str):
        super().__init__()

        self.mode = mode
        self.config = config
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                # transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        self.prepare_data()

    def prepare_data(self):
        # self.dataset, self.val_dataset = torch.utils.data.random_split(
        #         train_dataset, [45000, 5000]
        #     )
        if self.mode == "train":
            self.data_split = "train"

            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.datadir,
                train=True,
                download=True,
                transform=self.transform,
            )

            self.dataset = Subset(train_dataset,CIFAR_TRAIN_IDXS)
        elif self.mode == "val":
            self.data_split = "val"

            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.datadir,
                train=True,
                download=True,
                transform=self.test_transform,
            )

            self.dataset = Subset(train_dataset,CIFAR_VAL_IDXS)

        elif self.mode == "test":
            # add data_split
            self.data_split = "test"
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.config.datadir,
                train=False,
                download=False,
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        kspace_key = "sc_kspace"
        target_key = "recon_esc"

        kspace_arr = fftc(image)
        target_arr = image
        sampled_indices = (0, -1)

        parameters = {
            kspace_key: kspace_arr,
            target_key: target_arr,
            "sampled_indices": sampled_indices,
            "data_split": self.data_split,
            "dataset": self.config.dataset,
            "max_value": kspace_arr.abs().max(),
            "label": dict(cifar10=torch.tensor([label]).long()),
        }

        return parameters
