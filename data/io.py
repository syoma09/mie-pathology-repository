#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import ImageDraw

from data.svs import SVS


def save_thumbnail(svs: SVS, path):
    img, vtx = svs.get_thumbnail((1024, 1024))
    # Convert to list of tuple
    vtx = [(x, y) for [x, y] in vtx]

    # print(img.size)
    # print(vtx)

    draw = ImageDraw.Draw(img)
    # draw.polygon(vtx, fill=(0, 0, 0), outline=(0, 0, 0))
    # draw.polygon(vtx, outline=(0, 255, 0))

    x0, y0 = vtx[0]
    for x, y in vtx:
        print(x, y)
        # draw.point((x, y), fill=(0, 255, 0))
        draw.line([(x0, y0), (x, y)], fill=(0, 0, 255), width=3)
        x0, y0 = x, y

    img.save(path)
