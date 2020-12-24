#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from joblib import Parallel, delayed

from data.svs import SVS


def save_patches(path: Path, base, size, stride):
    """

    :param path:    Path to image.svs
    :param base:    Base string of output file name
    :param size:    Patch size
    :param stride:  Patch stride
    :return:        None
    """

    svs = SVS(path)

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        # print(p0, p1)
        img, mask = svs.extract_img_mask(p0, size)

        if np.sum(mask) == size[0] * size[1] * 255:
            print(i, np.sum(mask))

            img.save(str(base) + f"{i:08}img.png")
            # Image.fromarray(mask).save(str(base) + f"{i:08}mask.png")


def main():
    src = Path("~/data/mie-ortho/pathology/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(src / "list.csv")
    print(df)

    (dst / "0").mkdir(exist_ok=True)
    (dst / "1").mkdir(exist_ok=True)

    args = []
    for _, row in df.iterrows():
        subject = row['number']
        survive = row['survival']

        print(subject)

        # print(svs.image.slide.dimensions)

        # save_thumbnail(svs, str(dst / f"test-{subject}.jpg"))
        # save_thumbnail(svs, f"test-{subject}.jpg")

        path = src / subject / "image.svs"
        base = dst / str(survive) / f"{subject}_"
        size = 256, 256
        stride = size
        args.append((path, base, size, stride))

        # # Serial execution
        # save_patches(path, base, size=size, stride=stride)

    # Parallel execution
    Parallel(n_jobs=16)([
        delayed(save_patches)(path, base, size, stride)
        for path, base, size, stride in args
    ])


if __name__ == '__main__':
    main()
