#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from PIL import Image

from aipatho.metrics.label import TimeToLabelConverter
from aipatho.svs import TumorMasking, save_patches


# # To avoid "OSError: image file is truncated"
# ImageFile.LOAD_TRUNCATED_IMAGES = True


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: Path,
                 annotations: pd.DataFrame,
                 transform,
                 labeler: TimeToLabelConverter
                 ):
        super(PatchDataset, self).__init__()

        self.transform = transform
        # torchvision.transforms.Compose([
        #     # torchvision.transforms.Resize((224, 224)),
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        self.labeler = labeler

        self._dataset = [
            (path, label)  # Same label for one subject
            for subject, label in annotations
            for path in (root / subject).iterdir()
            if path.suffix not in ['.csv', '']
        ]
        # Random shuffle
        random.shuffle(self._dataset)

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
        return len(self._dataset)

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
        # img = self.data[item, :, :, :].view(3, 32, 32)
        path, label = self._dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # img = torchvision.transforms.functional.to_tensor(img)

        # Normalize
        # label /= 90.
        label = self.labeler(label)
        label = torch.tensor(label, dtype=torch.float)

        return img, label


class PatchCLDataset(PatchDataset):
    def __init__(self,
                 root: Path,
                 annotations: pd.DataFrame,
                 transform,
                 labeler: TimeToLabelConverter
                 ):
        super(PatchCLDataset, self).__init__(root, annotations, transform, labeler)

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor, int):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """

        path, _ = self._dataset[item]
        img = Image.open(path).convert('RGB')

        # ToDo: Check how true_class is used?
        true_class = torch.zeros(2, dtype=torch.float)
        true_class[1] = 1
        # if ("not" in str(path)):
        #     true_class[0] = 1
        # else:
        #     true_class[1] = 1

        return self.transform(img), self.transform(img), true_class
        # return self.transform(img), self.transform(img)


def load_annotation(path: Path) -> dict:
    # Load annotations
    annotation = {
        'train': [], 'valid': [], 'test': [], 'IGNORE': []
    }

    for _, row in pd.read_csv(path).iterrows():
        annotation[
            # Switch train/valid by tvt-column value (0: train, 1: valid) #tvt列の値（0：train、1：valid）でtrain/validを決定
            ['train', 'valid', 'test', 'IGNORE'][int(row['tvt'])]
        ].append(
            # (row['number'], row['OS'])              # Append annotation tuple
            (row['number'], row['survival time'])   # Append annotation tuple
            # (row['number'], row['survival time'], row['OS'])
            # (row['number'])
        )

    return annotation


def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size: (int, int),
        stride: (int, int),
        resize: (int, int) = (256, 256),
        index: int = None,
        region: int = None,
        target: TumorMasking = TumorMasking.FULL,
):
    # Load annotation
    df = pd.read_csv(annotation)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']
        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Subject #{number} already exists. Skip.")
            continue

        path_svs = src / f"{number}.svs"
        path_xml = src / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue
        base = subject_dir / 'patch'

        args.append((path_svs, path_xml, base))
        # # Serial execution
        # save_patches(
        #     path_svs, path_xml, base,
        #     size=size, stride=stride, resize=resize, index=index, region=region,
        #     target=target
        # )

    # Approx., 1 thread use 20GB
    # n_jobs = int(mem_total / 20)
    n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize, index, region, target)
        for path_svs, path_xml, base in args
    ])
