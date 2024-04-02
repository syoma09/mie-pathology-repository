#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from aipatho.svs import SVS


def main():
    src = Path("~/workspace/mie/pathology/_data/").expanduser()

    df = pd.read_csv(src / "survival_2dfs_v2.csv")
    print(df)

    for _, row in df.iterrows():
        subject = row['number']

        svs = SVS(
            src / "svs" / f"{subject}.svs",
            src / "xml" / f"{subject}.xml"
        )

        print(f"Subject-{subject}: {svs.image.slide.dimensions}")


if __name__ == '__main__':
    main()
