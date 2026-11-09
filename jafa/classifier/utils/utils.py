import argparse
import re
from typing import Sequence, Tuple

import numpy as np
import torch
import yaml
from joblib import dump
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm


def string_to_list(input_data):
    """Convert string or list to list format."""
    if isinstance(input_data, str):
        return list(input_data.split(","))
    if isinstance(input_data, list):
        return input_data
    raise ValueError(f"Input must be a string or list, got {type(input_data)}")


def get_config(config_path: str) -> argparse.ArgumentParser:
    with open(config_path, mode="r") as fp:
        config = yaml.safe_load(fp)
    return argparse.Namespace(**config)


def get_sampler_weights(dataset, save_filename):
    Y_tr = []
    for i in tqdm(range(len(dataset))):
        label = sum(list(dataset[i]["label"].values())).item()
        Y_tr.append(label)
    Y_tr = np.array(Y_tr).astype(np.long)
    class_sample_count = np.array(
        [len(np.where(Y_tr == t)[0]) for t in np.unique(Y_tr)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in Y_tr])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    dump(sampler, save_filename)


def extract_accuracy_from_filename(filename):
    """
    Extract balanced accuracy from checkpoint filename.
    Expected format: *balanced_accuracy=0.XXX.ckpt
    """
    match = re.search(r"balanced_accuracy=(\\d+\\.\\d+)\\.ckpt$", filename)
    if match:
        return float(match.group(1))
    return 0.0
