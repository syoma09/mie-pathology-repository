#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pathlib import Path

from cnn.utils import PatchDataset

# np.random.seed(1234)
# torch.manual_seed(1234)


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = Path("~/data/_out/").expanduser()

    epochs = 10000
    batch_size = 64    # Requires 19 GiB VRAM

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(root / 'train'), batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(root / 'valid'), batch_size=16
    )

    '''
    モデルの構築
    '''
    model = torchvision.models.resnet152(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    print(model)

    # Replace FC layer
    num_features = model.fc.in_features
    # print(num_features)  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2, bias=True),
        nn.Softmax(dim=1)
    )
    # Use GPU
    model = model.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    metrics = {}

    tensorboard = SummaryWriter(log_dir='./logs')
    for epoch in range(epochs):
        print(f"Epoch [{epoch:04}/{epochs:04}]:")

        # Switch to training mode
        model.train()
        for batch, (x, y_true) in enumerate(train_loader):
            # break
            x, y_true = x.cuda(), y_true.cuda()

            y_pred = model(x)   # Forward
            # print("yp:", y_pred)

            loss = criterion(y_pred, y_true)  # Calculate training loss
            loss.backward()     # Backward propagation
            optimizer.step()    # Update parameters

            # Logging
            # print("\033[2K\033[G")  # Clean current line

            progress = ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            print(f"\r  Batch({batch:6}/{len(train_loader):6})[{progress}]: {loss.item():.4}", end="")
            tensorboard.add_scalar('train_loss', loss.item(), epoch * batch_size + batch)

        # Switch to evaluation mode
        model.eval()
        # On training data

        # Initialize validation metric values
        valid_metrics = {key: 0 for key in ['loss'] + list(metrics.keys())}
        # Calculate validation metrics
        for x, y_true in valid_loader:
            x, y_true = x.cuda(), y_true.cuda()
            y_pred = model(x)  # Prediction

            loss = criterion(y_pred, y_true)  # Calculate validation loss
            valid_metrics['loss'] += loss.item() / len(valid_loader)
            for key, func in metrics.items():
                valid_metrics[key] += func(y_pred, y_true) / len(valid_loader)

        # Save to tensorboard
        for key, value in valid_metrics.items():
            print(f"valid_{key}: {value}")
            tensorboard.add_scalar('valid_' + key, value, epoch)


if __name__ == '__main__':
    main()
