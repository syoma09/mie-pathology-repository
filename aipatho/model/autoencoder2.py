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
    
    """#2048次元の特徴ベクトルを生成するように変更、デコーダの受けとりも。　viewの部分が分からない
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        # xはここで512次元の特徴マップである
        x = x.view(x.size(0), -1, 1, 1)  # ここで形状を変更しているが、次元数は変わらない(??)
        x = self.dec(x)

        return x

    @staticmethod
    def encoder() -> nn.Sequential:
        net = torchvision.models.resnet18()
        layers = list(net.children())[:-2]  # 最後の全結合層とAdaptiveAvgPool2dを除く
        
        # ここで512次元の出力を2048次元に変更する
        layers.append(nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(2048))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 1x1にプールして2048次元の特徴ベクトルを得る
        
        return nn.Sequential(*layers)

    @staticmethod
    def decoder() -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )"""

