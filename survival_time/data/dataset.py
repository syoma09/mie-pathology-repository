#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from aipatho.svs import save_patches, TumorMasking


def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        resize: (int, int) = (256, 256),
        index: int = None, region: int = None,
        target: TumorMasking = TumorMasking.FULL
):
    # Load annotation
    df = pd.read_csv(annotation)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']
        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Subject #{number} already exists. Skip.")
            continue

        path_svs = src / f"{number}.svs"
        path_xml = src / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue
        base = subject_dir / 'patch'

        args.append((path_svs, path_xml, base))
        # # Serial execution
        # save_patches(
        #     path_svs, path_xml, base,
        #     size=size, stride=stride, resize=resize, index=index, region=region,
        #     target=target
        # )

    # Approx., 1 thread use 20GB
    # n_jobs = int(mem_total / 20)
    n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize, index, region, target)
        for path_svs, path_xml, base in args
    ])
