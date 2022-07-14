#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw, Image

from data.svs import SVS


def save_thumbnail(svs: SVS, path, index: int = None, region: int = None):
    img, zoom = svs.thumbnail((1024, 1024))
    annots = svs.annotation.vertices(index, region, zoom)

    draw = ImageDraw.Draw(img)
    # draw.polygon(vtx, fill=(0, 0, 0), outline=(0, 0, 0))
    # draw.polygon(vtx, outline=(0, 255, 0))

    # TODO: Extracting 2nd+ annotation only
    # for _, annot in annots[-1:]:
    for _, annot in annots:
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
    dataset = Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12")
    dst = Path("~/data/_out/").expanduser()

    df = pd.read_csv(Path(
        "../_data/20220610_3os.csv"
    ).expanduser())
    print(df)

    for _, row in df.iterrows():
        subject = row['number']

        # if (dst / f"test-{subject}.jpg").exists():
        #     continue

        svs = SVS(
            dataset / f"{subject}.svs",
            dataset / f"{subject}.xml"
        )

        save_thumbnail(svs, str(dst / f"{subject}.jpg"))
        print(svs.image.slide.dimensions)


if __name__ == '__main__':
    main()
