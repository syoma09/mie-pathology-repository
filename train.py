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
        PatchDataset(root), batch_size=batch_size, shuffle=True
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

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tensorboard = SummaryWriter(log_dir='./logs')
    for epoch in range(epochs):
        print(f"Epoch [{epoch:04}/{epochs:04}]:")
        # Switch model to training mode
        model.train()

        for batch, (x, y_true) in enumerate(train_loader):
            x, y_true = x.cuda(), y_true.cuda()

            y_pred = model(x)   # Forward
            # print("yp:", y_pred)

            loss = criterion(y_pred, y_true)  # Calculate training loss
            print("  ", loss.item())
            tensorboard.add_scalar('train_loss', loss.item(), epoch * batch_size + batch)

            loss.backward()     # Backward propagation
            optimizer.step()    # Update parameters

        # # Switch to evaluation model
        # model.eval()
        # for x, y_true in valid_loader:
        #     x, y_true = x.cuda(), y_true.cuda()
        #     y_pred = model(x)  # Prediction
        #
        #     loss = criterion(y_pred, y_true)  # Calculate validation loss
        #     valid_loss += loss.item()


if __name__ == '__main__':
    main()
