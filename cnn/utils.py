#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random
from pathlib import Path

import torch.utils.data
import torchvision
from PIL import Image
from PIL import ImageOps


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        super(PatchDataset, self).__init__()

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0.5, 0.5)
        ])

        # self.classes = [d for d in root.iterdir()]
        # print(self.classes)
        self.classes = ["0", "1"]

        self.files = []
        ptn = re.compile(r"^.+57-10.+\.png$")
        for cls, name in enumerate(self.classes):
            self.files += [
                (f, cls) for f in (root / name).iterdir()
                if ptn.match(str(f))
            ]
            print(name, len(self.files))

        random.shuffle(self.files)  # Random shuffle

        # self.files = self.files[:5120]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """

        if item > len(self):
            item %= len(self)

        path, label = self.files[item]
        img = Image.open(path).convert('RGB')

        # Apply image pre-processing
        img = self.transform(ImageOps.mirror(img))   # / 255.
        # print(img.shape)

        # label = to1hot(label, 2)

        return img, label


def to1hot(cls, num):
    """

    :param cls: Class index
    :param num: Number of classes
    :return:    1-Hot vector encoded class
    """

    result = [0] * num
    result[cls] = 1

    return torch.tensor(result, dtype=torch.float)
