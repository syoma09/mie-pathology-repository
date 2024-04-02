#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import pandas as pd



def get_dataset_root_not_path(patch_size, stride, index):
    """
    :return:    Dataset root Path() object
    """

    # # Home directory
    # return Path("~/data/_out/mie-pathology/").expanduser()

    # Local SSD Cache
    path = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology'
    path /= "survival_p{}_s{}_not_i{}".format(
        f"{patch_size[0]}x{patch_size[1]}",
        f"{stride[0]}x{stride[1]}",
        f"{index}"
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
        ].append(
            # (row['number'], row['OS'])              # Append annotation tuple
            (row['number'], row['survival time'])   # Append annotation tuple
            # (row['number'], row['survival time'], row['OS'])
            # (row['number'])
        )
        
    return annotation
