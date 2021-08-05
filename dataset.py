#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data.svs import save_patches


# Check RAM capacity
with open('/proc/meminfo', 'r') as f:
    mem_total_str = [line for line in f.readlines() if line.startswith('MemTotal')]
    mem_total = re.findall(r'[0-9]+', mem_total_str[0])[0]
    mem_total = int(mem_total) / 1024 / 1024  # kB -> mB -> gB
    mem_total -= 4  # Keep 4GB for system
    print(mem_total)

# Approx., 1 thread use 20GB
n_jobs = int(mem_total / 20)
print(f'Process in {n_jobs} threads.')


def create():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()
    dst = Path("~/data/_out/mie-pathology/").expanduser()

    # Load annotation
    df = pd.read_csv(src / "AIpatho.csv")
    print(df)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']

        subject_dir = dst / number
        # if subject_dir.exists():
        #     print(f"Ignore {number}")
        #     continue
        #     # shutil.rmtree(subject_dir)
        # subject_dir.mkdir(parents=True, exist_ok=True)

        path_svs = src / "svs" / f"{number}.svs"
        path_xml = src / "xml" / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue

        base = dst / f"{number}" / "s{}_st{}_e{}_et{}_".format(
            subject['survival'],
            subject['survival time'],
            subject['event'],
            subject['event time']
        )
        size = 512, 512
        stride = size
        resize = 256, 256
        args.append((path_svs, path_xml, base, size, stride, resize))

        # # Serial execution
        # save_patches(path_svs, path_xml, base, size=size, stride=stride)

    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize)
        for path_svs, path_xml, base, size, stride, resize in args
    ])


if __name__ == '__main__':
    create()
