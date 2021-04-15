#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from joblib import Parallel, delayed

from data.svs import SVS


def create_time():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()

    with open(src / 'survival_time.yml', 'r') as f:
        dataset = yaml.safe_load(f)
        print(dataset)

    time_periods = [
        lambda t:       t < 12,
        # lambda t: 12 <= t < 24,
        # lambda t: 24 <= t < 36,
        lambda t: 12 <= t < 36,
        lambda t: 36 <= t
    ]

    for use, subjects in dataset.items():
        print(f"Processing \"{use}\" dataset.")

        for subject in subjects:
            with open(src / subject / 'data.yml') as f:
                yml = yaml.safe_load(f)

            s_time = float(yml['annotation']['survival']['time'])
            # Convert time to class
            for i, period in enumerate(time_periods):
                if period(s_time):
                    print(f'Subject{subject}: {i} ({s_time})')
                    break


if __name__ == '__main__':
    create_time()
