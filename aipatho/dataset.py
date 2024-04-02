#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image, ImageFile


# # To avoid "OSError: image file is truncated"
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class TimeToLabelConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: float) -> int:
        raise NotImplementedError


class TimeToTime(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return x


class TimeToFixed3(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x < 13:
            return 0
        elif x < 34:
            return 1
        else:   # elif x < 67:
            return 2


class TimeToFixed4(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x <= 11:
            return 0
        elif 11 < x <= 22:
            return 1
        elif 22 < x <= 33:
            return 2
        else:   # elif 33 < x <= 44:
            return 3


class TimeToFixed34(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x < 11:
            return 0
        elif x < 22:
            return 1
        elif x < 33:
            return 2
        elif x < 44:
            return 3
        elif x < 44:
            return 4
        elif x < 36:
            return 5
        elif x < 42:
            return 6
        elif x < 48:
            return 7
        # ToDo: Where is 8 ~ 10
        elif x < 24:
            return 11
        elif x < 26:
            return 12
        elif x < 28:
            return 13
        elif x < 30:
            return 14
        elif x < 32:
            return 15
        elif x < 34:
            return 16
        elif x < 36:
            return 17
        elif x < 38:
            return 18
        elif x < 40:
            return 19
        elif x < 42:
            return 20
        elif x < 44:
            return 21
        elif x < 46:
            return 22
        elif x < 48:
            return 23
        elif x < 50:
            return 24
        elif x < 52:
            return 25
        elif x < 54:
            return 26
        elif x < 56:
            return 27
        elif x < 58:
            return 28
        elif x < 60:
            return 29
        elif x < 62:
            return 30
        elif x < 64:
            return 31
        elif x < 66:
            return 32
        else:   # elif x < 68:
            return 33


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: Path,
                 annotations: pd.DataFrame,
                 labeler: TimeToLabelConverter
                 ):
        super(PatchDataset, self).__init__()

        self.labeler = labeler
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.__dataset = [
            (path, label)  # Same label for one subject
            for subject, label in annotations
            for path in (root / subject).iterdir()
            if path.suffix not in ['.csv', '']
        ]
        # Random shuffle
        random.shuffle(self.__dataset)

        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

    # def __str__(self) -> str:
    #     return "\n".join([
    #         f"PatchDataset",
    #         f"  # patch : {len(self.__dataset)}",
    #         f"  # of 0  : {len([l for _, l in self.__dataset if l <= 11])}",
    #         f"  # of 1  : {len([l for _, l in self.__dataset if 11 < l <= 22])}",
    #         f"  # of 2  : {len([l for _, l in self.__dataset if 22 < l <= 33])}",
    #         f"  # of 3  : {len([l for _, l in self.__dataset if 33 < l <= 44])}",
    #         f"  subjects: {sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset]))}",
    #     ])

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
        # img = self.data[item, :, :, :].view(3, 32, 32)
        path, label = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # img = torchvision.transforms.functional.to_tensor(img)

        # Normalize
        # label /= 90.
        label = self.labeler(label)
        label = torch.tensor(label, dtype=torch.float)

        return img, label
