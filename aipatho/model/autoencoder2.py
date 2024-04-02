#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchvision


class AutoEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = self.encoder()
        self.dec = self.decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        # print("Encoder output:", x.shape)

        # x = x.view(-1, self.num_flat_features(x))
        # print("Flattened     :", x.shape)

        x = self.dec(x)
        # print("Decoder output:", x.shape)

        return x

    @staticmethod
    def encoder() -> nn.Sequential:
        # input_size = 3 * 256 * 256
        net = torchvision.models.resnet18()
        layers = list(net.children())[:-1]

        # model = torchvision.models.resnet152(pretrained=False)
        # model = torchvision.models.resnet152(pretrained=True)

        return nn.Sequential(*layers)

    @staticmethod
    def decoder() -> nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    @staticmethod
    def num_flat_features(x: torch.Tensor) -> int:
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
