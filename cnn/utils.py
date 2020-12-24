#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import torch.utils.data
import torchvision
from PIL import Image


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path):
        super(PatchDataset, self).__init__()

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

        self.classes = [d for d in root.iterdir()]
        # print(self.classes)

        self.files = []
        for cls in self.classes:
            self.files.append(
                [f for f in (root / cls).iterdir()]
            )

        self.__len = sum(len(data) for data in self.files)

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
        path = self.files[0][0]
        img = Image.open(path).convert('RGB')
        label = 0

        # Apply image pre-processing
        img = self.transform(img)
        # print(img.shape)

        # label = torch.scalar_tensor(label, dtype=torch.int64)
        label = to1hot(label, 2)

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
