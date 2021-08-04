#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.transforms
import torchvision.transforms.functional
import skimage.io
import skimage.transform
from PIL import Image

from data.svs import SVS
from cnn.metrics import ConfusionMatrix
from survival import load_annotation, PatchDataset, create_model


# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


def generate_batch(batch_size, svs: SVS, size, stride, resize=None):
    batch = []

    for i, (p0, p1) in enumerate(svs.patches(size=size, stride=stride)):
        mask = svs.crop_mask(p0, size)
        # Check if the selected region is completely covered by mask
        if np.sum(mask) < size[0] * size[1] * 255:
            continue

        # Process image
        img = svs.crop_img(p0, size)
        if resize is not None:
            img = img.resize(resize)

        # RGBA -> RGB
        img = img.convert('RGB')
        batch.append((img, p0, p1))

        if len(batch) % batch_size == 0:
            yield batch     # Return batch
            batch = []      # Empty


def process_subject(model, svs: SVS, size, stride, resize=None):
    # Calculate image zoom ratio
    thumb, _ = svs.get_thumbnail((2048, 2048))
    mask = np.zeros(shape=thumb.size)
    # Expect: ratio[0] == ratio[1]
    ratio = (
        thumb.size[0] / svs.image.slide.dimensions[0],
        thumb.size[1] / svs.image.slide.dimensions[1]
    )
    print("Ratio:", ratio)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    c = 0
    for batch in generate_batch(
            batch_size=64, svs=svs,
            size=size, stride=stride, resize=resize
    ):
        print(c)
        c += 1

        imgs = [
            Variable(transforms(img), requires_grad=True)
            for img, _, _ in batch
        ]
        imgs = torch.stack(imgs).to(device)

        # Run prediction
        y_pred = np.argmax(
            model(imgs).to('cpu').detach().numpy(),
            axis=1
        )

        for (_, p0, p1), pred in zip(batch, y_pred):
            p0 = int(p0[0] * ratio[0]), int(p0[1] * ratio[1])
            p1 = int(p1[0] * ratio[0]), int(p1[1] * ratio[1])

            mask[p0[0]:p1[0], p0[1]:p1[1]] = 1

    return mask.T


def main():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()

    size = 1024, 1024
    stride = size
    resize = 256, 256

    target = 'cls'
    # target = '2dfs'
    # target = '3os'
    dataset = "valid"

    # Load annotations
    annotation = load_annotation(Path(
        f"~/workspace/mie-pathology/_data/survival_{target}.csv"
    ).expanduser())

    # Load trained model
    model = create_model().to(device)
    model_path = Path("~/data/_out/mie-pathology/").expanduser()
    model_path /= "20210803_091002/model00036.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for subject, label in annotation[dataset]:
        print(subject)

        path_svs = src / "svs" / f"{subject}.svs"
        path_xml = src / "xml" / f"{subject}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue
        svs = SVS(
            path_svs,
            annotation=path_xml
        )

        mask = process_subject(
            model=model, svs=svs, size=size, stride=stride, resize=resize
        )

        skimage.io.imsave(
            f"mask_{subject}.png", mask
        )
        break


if __name__ == '__main__':
    main()
