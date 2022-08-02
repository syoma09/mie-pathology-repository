#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, Image

from data.svs import SVS


def save_thumbnail(svs: SVS, hmap: Image, path, index: int = None, region: int = None):
    img, zoom = svs.thumbnail((hmap.width, hmap.height))
    annots = svs.annotation.vertices(index, region, zoom)

    # Resize to pathology image size. There may be few pixels size mismatch.
    hmap = hmap.resize(img.size)
    hmap = hmap.convert("RGB")

    """Alpha blending"""
    # hmap = np.array(hmap)
    # hmap[:, :, [0, 2]] = 0
    # hmap = Image.fromarray(hmap)
    # img = Image.blend(img, hmap, alpha=0.25)

    """Partial alpha blending"""
    # img = np.array(img)
    # hmap = np.array(hmap)
    # xs, ys, cs = np.where(hmap > 0)
    # xs, ys, cs = xs[1::3], ys[1::3], cs[1::3]
    # for i, j, c in zip(xs, ys, cs):
    #     img[i, j, 0] = 0
    #     img[i, j, 2] = 0
    #     img[i, j, c] = min(255, int(img[i, j, c] * 0.25 + hmap[i, j, c] * 0.75))
    # img = Image.fromarray(img)

    """Shadowing"""
    # img = np.array(img)
    # hmap = np.array(hmap)
    # # pts = np.where(hmap <= 0)
    # pts = np.where(hmap <= 127)
    # img[pts] = img[pts] // 2
    # img = Image.fromarray(img)

    """Shadow-fade"""
    img = np.array(img).astype(float)
    img = img // 2
    hmap = np.array(hmap).astype(float)
    xs, ys, cs = np.where(hmap > 0)
    for i, j, c in zip(xs, ys, cs):
        # img[i, j, 0] = 0
        # img[i, j, 2] = 0
        img[i, j, c] = min(255, int(img[i, j, c] * (1.0 + hmap[i, j, c] / 255.)))
    img = Image.fromarray(img.astype(np.uint8))

    """Overwrite"""
    # img = np.array(img)
    # hmap = np.array(hmap)
    # xs, ys, cs = np.where(hmap > 0)
    # pts = xs[1::3], ys[1::3], cs[1::3]
    # img[pts] = hmap[pts]
    # img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    # draw.polygon(vtx, fill=(0, 0, 0), outline=(0, 0, 0))
    # draw.polygon(vtx, outline=(0, 255, 0))

    # TODO: Extracting 2nd+ annotation only
    for _, annot in annots[-1:]:
        for _, region in annot:
            vtx = [(x, y) for [x, y] in region]
            # print(vtx)

            x0, y0 = vtx[0]
            for x, y in vtx:
                # print(x, y)
                # draw.point((x, y), fill=(0, 255, 0))
                draw.line([(x0, y0), (x, y)], fill=(0, 0, 255), width=3)
                x0, y0 = x, y

    img.save(path)

    # msk = svs.mask(annots[1:])
    # msk = msk.resize(img.size)

    # print(img.size, msk.size)
    # msk = np.array(msk, dtype=np.uint8) * 255
    # print(msk.dtype)
    # Image.fromarray(msk).save(path)


def main():
    # Input
    dataset = Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12")
    # heatmap = Path("~/data/_out/mie-pathology/heatmap/3-OS").expanduser()
    heatmap = Path("~/data/_out/mie-pathology/heatmap/3-MFS").expanduser()
    # Output
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(Path(
        "../_data/20220610_3os.csv"
    ).expanduser())
    print(df)

    for _, row in df.iterrows():
        subject = row['number']

        if not (heatmap / f"{subject}.jpg").exists():
            continue

        print(f"Processing... {subject}")
        hmap = Image.open(str(heatmap / f"{subject}.jpg"))
        svs = SVS(
            dataset / f"{subject}.svs",
            dataset / f"{subject}.xml"
        )

        save_thumbnail(svs, hmap, str(dst / f"{subject}.jpg"))


if __name__ == '__main__':
    main()
