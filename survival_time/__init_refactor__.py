#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image, ImageOps


def get_dataset_root_path(patch_size, stride):
    """
    :return:    Dataset root Path() object
    """

    # # Home directory
    # return Path("~/data/_out/mie-pathology/").expanduser()

    # Local SSD Cache
    path = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology'
    path /= "survival_p{}_s{}".format(
        f"{patch_size[0]}x{patch_size[1]}",
        f"{stride[0]}x{stride[1]}"
    )

    return path

def load_annotation(path: Path):
    # Load annotations
    annotation = {
        'train': [], 'valid': [], 'test': [], 'IGNORE': []
    }

    for _, row in pd.read_csv(path).iterrows():
        annotation[
            # Switch train/valid by tvt-column value (0: train, 1: valid)
            ['train', 'valid', 'test', 'IGNORE'][int(row['tvt'])]
    ].append((row['number'], row['OS']))     # Append annotation tuple
    # ].append((row['number'], row['label']))     # Append annotation tuple

    return annotation
