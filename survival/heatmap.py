#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from functools import cached_property

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.backends import cudnn
import torchvision.transforms
import torchvision.transforms.functional
from PIL import Image

from data.svs import SVS
from survival import load_annotation, get_transform, create_model


# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


class StrideHeatmap(object):
    def __init__(
            self,
            path: Path,
            dimensions: (int, int),
            size: (int, int), stride: (int, int)
    ):
        """

        :param path:        Path to np.memmap
        :param dimensions:  Original image size
        :param size:        Patch size
        :param stride:      Patch stride parameter
        """

        self.__size = size
        self.__stride = stride

        shape = dimensions[0] // stride[0], dimensions[1] // stride[1]
        print(shape)

        self.__hmap = np.memmap(
            str(path.parent / "hmap.npy"), mode="w+",
            shape=shape, dtype=float
        )
        self.__hmap[:, :] = 0     # Initialize

    @property
    def size(self):
        return self.__size

    @property
    def stride(self):
        return self.__stride

    def __pixel2block(self, x, y):
        return (
            x // self.stride[0],
            y // self.stride[1]
        )

    def put(self, p: float, x: int, y: int, w: int, h: int):
        """

        :param p: Probability value
        :param x: Upper-left coordinate in pixel
        :param y: Lower-right coordinate in pixel
        :param w: Width in pixel
        :param h: Height in pixel
        :return:
        """
        x0, y0 = self.__pixel2block(x, y)
        x1, y1 = self.__pixel2block(x+w, y+h)
        self.__hmap[x0:x1, y0:y1] += 1.0 - p

    def normalized(self):
        norm = (self.size[0] / self.stride[0]) * (self.size[1] / self.stride[1])
        return self.__hmap[:, :] / norm
        # return self.__hmap[:, :] / np.max(self.__hmap[:, :])

    def save(self, path: Path):
        hmap = self.normalized()

        img = np.array(hmap * 255).astype(np.uint8)
        Image.fromarray(img.T).save(path)


class PatchPosDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        super(PatchPosDataset, self).__init__()

        self.__transform = get_transform()

        self.__dataset = []
        for _, row in df.iterrows():
            # x0, y0 = row['x'], row['y']
            # x1, y1 = x0 + row['width'], y0 + row['height']
            # path = row['path']
            self.__dataset.append((
                row['x'], row['y'], row['width'], row['height'], row['path']
            ))

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item: int):
        x, y, w, h, path = self.__dataset[item]

        img = self.__transform(
            Image.open(path).convert("RGB")
        )

        return x, y, w, h, img


# def generate_batch(batch_size, svs: SVS, size, stride, resize=None):
#     batch = []
#
#     for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
#         mask = svs.crop_mask(p0, size)
#         # Check if the selected region is completely covered by mask
#         if np.sum(mask) < size[0] * size[1] * 255:
#             continue
#
#         # Process image
#         img = svs.crop_img(p0, size)
#         if resize is not None:
#             img = img.resize(resize)
#
#         # RGBA -> RGB
#         img = img.convert('RGB')
#         batch.append((img, p0, p1))
#
#         if len(batch) % batch_size == 0:
#             yield batch     # Return batch
#             batch = []      # Empty


def process_subject(
        model,
        path: Path,     # Path to patch list
        dst: Path,
        svs: SVS,
        size: (int, int), stride: (int, int), resize=None
) -> Image:
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor()
    # ])
    data_loader = torch.utils.data.DataLoader(
        PatchPosDataset(pd.read_csv(path)),
        batch_size=128, shuffle=False, num_workers=os.cpu_count() // 2
    )

    # # Calculate image zoom ratio
    # thumb, _ = svs.thumbnail((2048, 2048))
    # mask = np.zeros(shape=thumb.size)
    # # Expect: ratio[0] == ratio[1]
    # ratio = (
    #     thumb.size[0] / svs.image.slide.dimensions[0],
    #     thumb.size[1] / svs.image.slide.dimensions[1]
    # )
    # print("Ratio:", ratio)

    # norm = (size[0] / stride[0]) * (size[1] / stride[1])
    # # 32bit float image
    # # hmap = Image.new(mode="F", size=svs.image.slide.dimensions, color=0)
    # hmap = np.memmap(
    #     str(path.parent / "hmap.npy"), mode="w+",
    #     shape=svs.image.slide.dimensions, dtype=float
    # )
    # hmap[:, :] = 0
    hmap = StrideHeatmap(dst, svs.image.slide.dimensions, size, stride)

    # patch_list = pd.read_csv(path)
    #
    # for i, row in patch_list.iterrows():
    #     if i % 50 == 0:
    #         print("---", i)
    #     x0, y0 = row['x'], row['y']
    #     x1, y1 = x0 + row['width'], y0 + row['height']
    #
    #     hmap.put(1.0, x0, y0, row['width'], row['height'])
    #     # hmap[x0:x1, y0:y1] += 1.0
    with torch.no_grad():
        for batch, (xs, ys, ws, hs, img) in enumerate(data_loader):
            print(f"Batch [{batch:3}/{len(data_loader):3}]")
            img = img.to(device)
            pred = model(img)

            pred = torch.argmax(pred, dim=1)
            for x, y, w, h, p in zip(xs, ys, ws, hs, pred):
                # print(x, y, w, h, float(p))
                hmap.put(float(p), int(x), int(y), int(w), int(h))

    print("Writing heatmap image.")
    hmap.save(dst / f"{path.parent.stem}.jpg")
    # Image.fromarray(hmap).save(str(path.parent / "hmap.jpg"))

    # c = 0
    # for batch in generate_batch(
    #         batch_size=64, svs=svs,
    #         size=size, stride=stride, resize=resize
    # ):
    #     print(c)
    #     c += 1
    #
    #     imgs = [
    #         Variable(transforms(img), requires_grad=True)
    #         for img, _, _ in batch
    #     ]
    #     imgs = torch.stack(imgs).to(device)
    #
    #     # Run prediction
    #     y_pred = np.argmax(
    #         model(imgs).to('cpu').detach().numpy(),
    #         axis=1
    #     )
    #
    #     for (_, p0, p1), pred in zip(batch, y_pred):
    #         p0 = int(p0[0] * ratio[0]), int(p0[1] * ratio[1])
    #         p1 = int(p1[0] * ratio[0]), int(p1[1] * ratio[1])
    #
    #         mask[p0[0]:p1[0], p0[1]:p1[1]] = pred
    #
    # return mask.T


def main():
    size = 1024, 1024
    stride = 128, 128
    resize = 256, 256

    dataset = "test"

    label, cv, exp, epoch = [
        ("20220610_3os", 0, "20220615_143124", 292),
        ("20220610_3os", 1, "20220615_143117", 185),
        ("20220610_3os", 2, "20220615_150709", 5),
        ("20220610_3os", 3, "20220616_110237", 200),
        ("20220610_3mfs", 0, "20220617_122426", 249),
        ("20220610_3mfs", 1, "20220616_183431", 45),
        ("20220610_3mfs", 2, "20220620_132834", 182),
        ("20220610_3mfs", 3, "20220620_132625", 224)
    ][7]
    # TODO: redo-0

    # Load annotations
    annotation = load_annotation(Path(
        f"~/workspace/mie/pathology/_data/{label}/cv{cv}.csv"
    ).expanduser())

    # Load trained model
    model = create_model().to(device)
    model_path = Path("~/data/_out/mie-pathology/").expanduser()
    model_path /= f"{exp}/model{epoch:05}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    hmap_path = model_path.parent / "heatmap"
    hmap_path.mkdir(parents=True, exist_ok=True)

    for subject, label in annotation[dataset]:
        print(subject)

        path_svs = Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12") / f"{subject}.svs"
        path_xml = Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12") / f"{subject}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue

        svs = SVS(path_svs, annotation=path_xml)

        process_subject(
            model=model,
            path=Path(
                f"~/cache/mie-pathology/survival_p{size[0]}x{size[1]}_s{stride[0]}x{stride[1]}/{subject}/patchlist.csv"
            ).expanduser(),
            dst=hmap_path,
            svs=svs, size=size, stride=stride, resize=resize
        )

        # break


if __name__ == '__main__':
    main()
