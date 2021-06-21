#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from data.svs import SVS
from data.io import save_thumbnail


def main():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(src / "survival_2dfs.csv")
    print(df)

    for _, row in df.iterrows():
        subject = row['number']

        svs = SVS(
            src / "svs" / f"{subject}.svs",
            src / "xml" / f"{subject}.xml"
        )

        save_thumbnail(svs, str(dst / f"test-{subject}.jpg"))


if __name__ == '__main__':
    main()
