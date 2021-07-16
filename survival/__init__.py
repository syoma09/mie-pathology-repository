#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from PIL import Image, ImageOps


def get_dataset_root_path(target):
    """

    :return:    Dataset root Path() object
    """

    # # Home directory
    # return Path("~/data/_out/mie-pathology/").expanduser()

    # Local SSD Cache
    return Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology' / f'survival_{target}'


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, annotations: list):
        """

        :param root:            Path to dataset root directory
        :param annotations:     List of (subject, label).
        """
        super(PatchDataset, self).__init__()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )   # Normalization for ImageNet pretrained model
            # torchvision.transforms.Normalize(0.5, 0.5)
        ])

        self.__dataset = []

        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
            ]
        # Random shuffle
        random.shuffle(self.__dataset)

        # self.__num_class = len(set(label for _, label in self.__dataset))
        self.__num_class = 2
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([l for _, l in self.__dataset if l == 0]))
        print('  # of 1  :', len([l for _, l in self.__dataset if l == 1]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """

        if item > len(self):
            item %= len(self)

        path, label = self.__dataset[item]
        img = Image.open(path).convert('RGB')

        # Apply image pre-processing
        img = self.transform(ImageOps.mirror(img))   # / 255.
        # print(img.shape)

        # # Single node output
        # target = torch.tensor([label], dtype=torch.float)
        # Convert to 1-Hot vector
        target = [0.0] * self.__num_class
        target[label] = 1.0
        target = torch.tensor(target, dtype=torch.float)

        return img, target


def create_model():
    """

    :return:    None
    """
    # # VGG16: 134M params
    # # model = torchvision.models.vgg16(pretrained=True)
    # model = torchvision.models.vgg16(pretrained=False)
    """VGG11Bn: """
    model = torchvision.models.vgg11_bn(pretrained=True)
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features, out_features=2, bias=True
    )

    # """ResNet152: 60M params"""
    # model = torchvision.models.resnet152(pretrained=False)
    # # model = torchvision.models.resnet152(pretrained=True)       # Too large -> Over-fitting
    # """ResNet50: 25M params"""
    # # model = torchvision.models.resnet50(pretrained=False)
    # model.fc = nn.Linear(model.fc.in_features, 2, bias=True)

    # # Inception-v3: 25M params
    # model = torchvision.models.inception_v3(pretrained=False)
    # # model.fc = nn.Linear(model.fc.in_features, 1, bias=True)
    # model.fc = nn.Linear(model.fc.in_features, 2, bias=True)

    # """DenseNet121: 58M params"""
    # model = torchvision.models.densenet121(pretrained=False)
    # model.classifier = nn.Linear(
    #     in_features=model.classifier.in_features, out_features=2, bias=True
    # )

    print(model)

    # num_params = sum(param.numel() for param in model.parameters())
    # print(f"{num_params} parameters")
    # print(f"{num_params // 1000}K parameters")
    # print(f"{num_params // 1000000}M parameters")

    # exit(0)
    return model


def load_annotation(path: Path):
    # Load annotations
    annotation = {
        'train': [], 'valid': [], 'test': [], 'IGNORE': []
    }

    for _, row in pd.read_csv(path).iterrows():
        annotation[
            # Switch train/valid by tvt-column value (0: train, 1: valid)
            ['train', 'valid', 'test', 'IGNORE'][int(row['tvt'])]
        ].append((row['number'], row['label']))     # Append annotation tuple

    return annotation
