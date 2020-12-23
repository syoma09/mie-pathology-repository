#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import yaml

from data.svs import SVS
from data.io import save_thumbnail


def main():
    root = Path("~/data/mie-ortho/pathology/").expanduser()
    df = pd.read_csv(root / "list.csv")

    print(df)
    for subject in df['number']:
        print(subject)

        svs = SVS(root / subject / "image.svs")
        print(svs.image.slide.dimensions)

        # save_thumbnail(svs, f"test-{subject}.jpg")
        print("Patches")
        size = 256, 256
        for i, (p0, p1) in enumerate(svs.patches(size=size, stride=(64, 64))):
            # print(p0, p1)
            img, mask = svs.extract_img_mask(p0, size)

            if np.sum(mask) > 0:
                print(i, np.sum(mask))
                base = Path(f"~/data/_out/{i:04}").expanduser()
                cv2.imwrite(str(base) + "img.jpg", img)
                cv2.imwrite(str(base) + "mask.jpg", mask)
        break


if __name__ == '__main__':
    main()
