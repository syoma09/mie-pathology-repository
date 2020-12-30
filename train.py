#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pathlib import Path

from cnn.utils import PatchDataset
from cnn.metrics import ConfusionMatrix

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
    model = torchvision.models.resnet152(pretrained=False)
    # model = torchvision.models.resnet50(pretrained=True)
    # print(model)

    # Replace FC layer
    num_features = model.fc.in_features
    # print(num_features)  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2, bias=True),
        # nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    # Use GPU
    model = model.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    tensorboard = SummaryWriter(log_dir='./logs')

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")

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

            # progress = ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            # print("\r  Batch({batch:6}/{len(train_loader):6})[{progress}]: {loss.item():.4}".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
            tensorboard.add_scalar(
                'train_loss', loss.item(), epoch * batch_size + batch
            )
        print('')

        # Switch to evaluation mode
        model.eval()
        # On training data

        # Initialize validation metric values
        metrics = {
            'loss': 0,
            'cmat': ConfusionMatrix(None, None)
        }
        # Calculate validation metrics
        for x, y_true in valid_loader:
            x, y_true = x.cuda(), y_true.cuda()
            y_pred = model(x)  # Prediction

            loss = criterion(y_pred, y_true)  # Calculate validation loss
            metrics['loss'] += loss.item() / len(valid_loader)
            metrics['cmat'] += ConfusionMatrix(y_pred, y_true)

        # Console write
        print("    valid_loss: {:3.3}".format(metrics['loss']))
        print("    valid_acc : {:3.3}".format(metrics['cmat'].accuracy()))
        print("    valid_f1  : {:3.3}".format(metrics['cmat'].f1()))
        # Write tensorboard
        tensorboard.add_scalar('valid_loss', metrics['loss'], epoch)
        tensorboard.add_scalar('valid_acc', metrics['cmat'].accuracy(), epoch)
        tensorboard.add_scalar('valid_f1', metrics['cmat'].f1(), epoch)


if __name__ == '__main__':
    main()
