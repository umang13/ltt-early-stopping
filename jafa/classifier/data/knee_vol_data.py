import argparse
from typing import Sequence, Dict
import gc

import torch
import h5py
import numpy as np

from torchvision import transforms

from .mr_vol_dataset import MRVolumeDataset


class KneeVolumeDataset(MRVolumeDataset):
    def __init__(
        self,
        config: argparse.ArgumentParser,
        mode: str,
        dev_mode: bool = False,
        transforms: transforms.Compose = None,
    ):
        super().__init__(config=config, mode=mode, dev_mode=dev_mode, transforms=transforms)
        self.config = config

    def get_volume(self, volume_id):
        vol_info = self.metadata[self.metadata.volume_id == volume_id]
        n_slices = len(vol_info)

        kspace_key = "sc_kspace"
        target_key = "recon_esc"

        # # Create transforms without ToTensor to avoid creating many tensors
        # # We'll apply ToTensor once at the end
        # transforms_no_totensor = None
        # totensor = None
        # if self.transforms is not None and hasattr(self.transforms, 'transforms'):
        #     # Extract ToTensor and create transforms without it
        #     transform_list = []
        #     for t in self.transforms.transforms:
        #         if isinstance(t, transforms.ToTensor):
        #             totensor = t
        #         else:
        #             transform_list.append(t)
        #     if transform_list:
        #         transforms_no_totensor = transforms.Compose(transform_list)

        # vol_kspace_list = []
        # vol_target_list = []
        vol_kspace_arr = (
            [None for _ in range(n_slices)]
            if not self.config.ignore_kspace
            else [np.nan] * n_slices
        )
        vol_target_arr = (
            [None for _ in range(n_slices)]
            if not self.config.ignore_image
            else [np.nan] * n_slices
        ) 

        for slice_idx in range(n_slices):
            info = vol_info[vol_info.slice_id == slice_idx].iloc[0]
            loc = info.location

            with h5py.File(loc) as f:
                kspace_arr = (
                    f[kspace_key][:] if not self.config.ignore_kspace else np.nan,
                )
                target_arr = (
                    f[target_key][:] if not self.config.ignore_image else np.nan,
                )
                # print(len(kspace_arr))
                target_arr, kspace_arr = self.apply_transforms(image=target_arr[0], kspace=kspace_arr[0])
                # print(kspace_arr.shape)
            # Apply transforms without ToTensor (only random flips)
            # if transforms_no_totensor is not None:
            #     if not self.config.ignore_image and target_arr is not np.nan:
            #         target_arr = transforms_no_totensor(target_arr)
            #     if not self.config.ignore_kspace and kspace_arr is not np.nan:
            #         stacked = np.stack([kspace_arr.real, kspace_arr.imag], axis=2)
            #         stacked = transforms_no_totensor(stacked)
            #         kspace_arr = stacked[..., 0] + 1j * stacked[..., 1]

            vol_kspace_arr[slice_idx] = kspace_arr
            vol_target_arr[slice_idx] = target_arr
        # print("vol_target_arr.shape", vol_target_arr[0].shape, "vol_target_arr length", len(vol_target_arr))
        # Concatenate as numpy arrays (memory efficient)
        # print(len(vol_kspace_arr), vol_kspace_arr[0].shape)
        vol_kspace_arr = np.stack(vol_kspace_arr, axis=0)
        # print("At dataset : ", vol_kspace_arr.shape)
        if not self.config.ignore_image:
            vol_target_arr = np.stack(vol_target_arr, axis=0)
        else:
            vol_target_arr = vol_target_arr[0]
        # Clear lists to free memory
        # del vol_kspace_list, vol_target_list
        # gc.collect()

        # # Apply ToTensor once to the entire volume (single tensor conversion)
        # if totensor is not None:
        #     if not self.config.ignore_kspace and vol_kspace_arr is not np.nan:
        #         # Handle complex kspace: stack real/imag, convert, then reconstruct
        #         stacked = np.stack([vol_kspace_arr.real, vol_kspace_arr.imag], axis=-1)
        #         stacked_tensor = totensor(stacked)
        #         vol_kspace_arr = stacked_tensor[..., 0] + 1j * stacked_tensor[..., 1]
        #     if not self.config.ignore_image and vol_target_arr is not np.nan:
        #         vol_target_arr = totensor(vol_target_arr)

        return {kspace_key: vol_kspace_arr, target_key: vol_target_arr}

    def parse_label(self, label_arr: Sequence[str]) -> Dict:
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")

        if "None" in label_arr:
            return dict(
                acl=torch.zeros(1).long(),
                mtear=torch.zeros(1).long(),
                abnormal=torch.zeros(1).long(),
            )

        new_labels = []

        for label in label_arr:
            if "ACL" in label:
                new_labels.append(1)
            elif "Meniscus Tear" in label:
                new_labels.append(2)
            else:
                new_labels.append(3)

        abnormal = 1.0 if 3 in new_labels else 0.0
        mtear = 1.0 if 2 in new_labels else 0.0
        acl = 1.0 if 1 in new_labels else 0.0

        return dict(
            acl=torch.Tensor([acl]).long(),
            mtear=torch.Tensor([mtear]).long(),
            abnormal=torch.Tensor([abnormal]).long(),
        )

    def get_volume_label(self, volume_id):
        vol_info = self.metadata[self.metadata.volume_id == volume_id]
        n_slices = len(vol_info)
        vol_labels = dict(
            acl=torch.tensor([0]),
            mtear=torch.tensor([0]),
            abnormal=torch.tensor([0]),
        )
        slice_labels = []
        for label_arr in vol_info.labels:
            slice_label = self.parse_label(label_arr=label_arr)
            slice_labels.append(slice_label)
            for key in slice_label:
                vol_labels[key] = max(vol_labels[key], slice_label[key])

        return vol_labels, slice_labels

    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        info = self.metadata_by_mode[self.mode].iloc[index]

        vol_inputs = self.get_volume(volume_id=info.volume_id)
        vol_label, slice_labels = self.get_volume_label(volume_id=info.volume_id)

        parameters = {
            **vol_inputs,
            "volume_id": info.volume_id,
            "slice_id": info.slice_id,
            "data_split": info.data_split,
            "dataset": info.dataset,
            "location": info.location,
            "max_value": info.max_value,
            "label": vol_label,
            "slice_labels": slice_labels,
        }

        return parameters
