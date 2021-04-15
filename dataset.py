#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from joblib import Parallel, delayed

from data.svs import SVS


def save_patches(path: Path, base, size, stride, resize=None):
    """

    :param path:    Path to image.svs
    :param base:    Base string of output file name
    :param size:    Patch size
    :param stride:  Patch stride
    :param resize:  Resize extracted patch
    :return:        None
    """

    svs = SVS(path)

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        patch_path = str(base) + f"{i:08}img.png"
        if Path(patch_path).exists():
            continue

        # print(p0, p1)
        img, mask = svs.extract_img_mask(p0, size)

        if np.sum(mask) == size[0] * size[1] * 255:
            if resize is not None:
                img = img.resize(resize)

            print(patch_path)
            img.save(patch_path)
            # Image.fromarray(mask).save(str(base) + f"{i:08}mask.png")


def create_survival():
    src = Path("~/data/mie-ortho/pathology/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(src / "list.csv")
    print(df)

    (dst / 'train' / '0').mkdir(parents=True, exist_ok=True)
    (dst / 'train' / '1').mkdir(parents=True, exist_ok=True)
    (dst / 'valid' / '0').mkdir(parents=True, exist_ok=True)
    (dst / 'valid' / '1').mkdir(parents=True, exist_ok=True)

    args = []
    for _, row in df.iterrows():
        subject = row['number']
        survive = row['survival']
        dataset = [
            False,      # use==0: Ignore
            'train',    # use==1: Use as training data
            'valid'     # use==2: Use as validation data
        ][int(row['use'])]

        if not dataset:
            continue

        print(subject)

        # print(svs.image.slide.dimensions)

        path = src / subject / "image.svs"
        base = dst / dataset / str(survive) / f"{subject}_"
        size = 512, 512
        stride = size
        resize = 256, 256
        args.append((path, base, size, stride, resize))

        # # Serial execution
        # save_patches(path, base, size=size, stride=stride)

    # Parallel execution
    Parallel(n_jobs=12)([
        delayed(save_patches)(path, base, size, stride, resize)
        for path, base, size, stride, resize in args
    ])


def create_time():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    with open(src / 'survival_time.yml', 'r') as f:
        dataset = yaml.safe_load(f)
        print(dataset)

    time_periods = [
        lambda t:       t < 12,
        lambda t: 12 <= t < 36,
        lambda t: 36 <= t
    ]
    for i, _ in enumerate(time_periods):
        (dst / 'train' / str(i)).mkdir(parents=True, exist_ok=True)
        (dst / 'valid' / str(i)).mkdir(parents=True, exist_ok=True)

    args = []
    for use, subjects in dataset.items():
        print(f"Processing \"{use}\" dataset.")

        for subject in subjects:
            with open(src / subject / 'data.yml') as f:
                yml = yaml.safe_load(f)

            s_time = float(yml['annotation']['survival']['time'])
            # Convert time to class
            for i, period in enumerate(time_periods):
                if period(s_time):
                    s_time = i
                    break

            # print(svs.image.slide.dimensions)

            path = src / subject / yml['image']['svs']
            base = dst / use / str(s_time) / f"{subject}_"
            size = 512, 512
            stride = size
            resize = 256, 256
            args.append((path, base, size, stride, resize))

            # # Serial execution
            # save_patches(path, base, size=size, stride=stride)

    # Parallel execution
    Parallel(n_jobs=12)([
        delayed(save_patches)(path, base, size, stride, resize)
        for path, base, size, stride, resize in args
    ])


if __name__ == '__main__':
    # create_survival()
    create_time()
