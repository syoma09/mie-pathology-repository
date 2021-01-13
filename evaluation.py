#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.backends import cudnn

from cnn.utils import PatchDataset
from cnn.metrics import ConfusionMatrix

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)


def main():
    root = Path("~/data/_out/").expanduser()

    batch_size = 512                    # 512 requires 9.7 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT

    # データ読み込み
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(root / 'valid'), batch_size=batch_size,
        num_workers=num_workers
    )

    '''
    モデルの構築
    '''
    # model = torchvision.models.resnet152(pretrained=False)
    model = torchvision.models.resnet152(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # print(model)

    # Replace FC layer
    num_features = model.fc.in_features
    # print(num_features)  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2, bias=True),
    )
    model = model.to(device)
    model.load_state_dict(torch.load(
        root / "20210108_111733model00022.pth",
        # root / "20210108_111733model00063.pth",
        map_location=device
    ))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    # Initialize validation metric values
    metrics = {
        'loss': 0.,
        'cmat': ConfusionMatrix(None, None)
    }
    # Calculate validation metrics
    with torch.no_grad():
        for i, (x, y_true) in enumerate(valid_loader):
            print("\r Valid ({:5}/{:5})".format(
                i, len(valid_loader)
            ), end='')
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)  # Prediction

            loss = criterion(y_pred, y_true)  # Calculate validation loss
            # print(loss.item())
            metrics['loss'] += loss.item() / len(valid_loader)
            metrics['cmat'] += ConfusionMatrix(y_pred, y_true)
    print("")

    # Console write
    print("    valid loss: {:3.3}".format(metrics['loss']))
    print("          acc : {:3.3}".format(metrics['cmat'].accuracy()))
    print("          f1  : {:3.3}".format(metrics['cmat'].f1()))
    print("        Matrix:")
    print(metrics['cmat'])


if __name__ == '__main__':
    main()
