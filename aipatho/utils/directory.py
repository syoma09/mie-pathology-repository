#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path

from aipatho.svs import TumorMasking

#ログを残すディレクトリ
def get_logdir(root: Path = Path("~/data/_out/mie-pathology/")) -> Path:
    log_root = root.expanduser().absolute() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)

    return log_root

#パッチ画像の保存先ディレクトリ
def get_cache_dir(patch: (int, int), stride: (int, int), target: TumorMasking) -> Path:
    """
    :return:    Dataset root Path() object
    """

    # # Home directory
    return Path("~/data/_out/mie-pathology/").expanduser()

    # Local SSD Cache
    """
    path = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology'
    path /= "survival_p{}_s{}_t{}".format(
        f"{patch[0]}x{patch[1]}",
        f"{stride[0]}x{stride[1]}",
        f"{int(target)}"
    )
    """
    # Create dataset if not exists
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path
