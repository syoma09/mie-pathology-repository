#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from data.svs import SVS
from data.io import save_thumbnail


def main():
    src = Path("~/data/mie-ortho/pathology/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(src / "list.csv")
    print(df)

    for _, row in df.iterrows():
        subject = row['number']

        svs = SVS(src / subject / 'image.svs')

        save_thumbnail(svs, str(dst / f"test-{subject}.jpg"))
        # save_thumbnail(svs, f"test-{subject}.jpg")


if __name__ == '__main__':
    main()
