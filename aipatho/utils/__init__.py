#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.utils.data

from aipatho.utils.directory import get_logdir


def to1hot(cls, num):
    """

    :param cls: Class index
    :param num: Number of classes
    :return:    1-Hot vector encoded class
    """

    result = [0] * num
    result[cls] = 1

    return torch.tensor(result, dtype=torch.float)
