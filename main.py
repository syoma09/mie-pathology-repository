#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import cv2
import yaml

from annotation import SVS


root = Path("~/data/mie-ortho/pathology").expanduser()
subject = "57-10"

with open(root / subject / "data.yml", "r") as f:
    yml = yaml.safe_load(f)
    print(yml)

# slide = OpenSlide(str(root / subject / yml["raw"]["svs"]))
# print("level_count: ", slide.level_count)
# print("dimensions : ", slide.dimensions)
# # img = slide.get_thumbnail((512, 512))
# img = slide.get_thumbnail(slide.dimensions)
# print(img)
# img.save("./test.jpg")

svs = SVS(root / subject / yml["raw"]["svs"])
# print(annot.Vertices)
img, vtx = svs.get_thumbnail((1024, 1024))

print(img.shape)
print(vtx)
print(np.max(vtx, axis=0))

for v in vtx:
    img[v[0]-1:v[0]+1, v[1]-1:v[1]+1, :] = [0, 255, 0]
# img[vtx] = [255, 0, 0]

cv2.imwrite("test.jpg", img)
