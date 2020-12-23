#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from data.svs import SVS


def save_thumbnail(svs: SVS, path):
    img, vtx = svs.get_thumbnail((1024, 1024))

    print(img.shape)
    print(vtx)
    print(np.max(vtx, axis=0))

    for v in vtx:
        img[v[0] - 1:v[0] + 1, v[1] - 1:v[1] + 1, :] = [0, 255, 0]
    # img[vtx] = [255, 0, 0]

    cv2.imwrite(path, img)
