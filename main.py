#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import cv2

from data.svs import SVS
from data.io import save_thumbnail


def save_patches(base, svs: SVS, size, stride):
    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        # print(p0, p1)
        img, mask = svs.extract_img_mask(p0, size)

        if np.sum(mask) == size[0] * size[1] * 255:
            print(i, np.sum(mask))

            cv2.imwrite(str(base) + f"{i:08}img.jpg", img)
            cv2.imwrite(str(base) + f"{i:08}mask.jpg", mask)


def main():
    src = Path("~/data/mie-ortho/pathology/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(src / "list.csv")
    print(df)

    (dst / "0").mkdir(exist_ok=True)
    (dst / "1").mkdir(exist_ok=True)

    for df_row in df.iterrows():
        subject = df_row['subject']
        survive = df_row['survival']

        print(subject)

        svs = SVS(src / subject / "image.svs")
        # print(svs.image.slide.dimensions)

        # save_thumbnail(svs, str(dst / f"test-{subject}.jpg"))
        # save_thumbnail(svs, f"test-{subject}.jpg")

        save_patches(
            str(dst / survive / f"{subject}_"),
            svs, size=(256, 256), stride=(256, 256)
        )

        break


if __name__ == '__main__':
    main()
