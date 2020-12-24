#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from pathlib import Path

from cnn.utils import PatchDataset

# np.random.seed(1234)
# torch.manual_seed(1234)


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = Path("~/data/_out/").expanduser()

    # # patch_dataset[z,x,y,]
    # patch_dataset = PatchDataset(root / "affect_data")
    # valid_dataset = PatchDataset(root / "valid_data")
    # print(len(patch_dataset))

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(root), batch_size=32, shuffle=True
    )
    # test_dataloader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=3, shuffle=False
    # )

    '''
    モデルの構築
    '''
    model = torchvision.models.resnet50(pretrained=True)
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

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch [{epoch:04}/{epochs:04}]:")
        # Switch model to training mode
        model.train()

        for x, y_true in train_loader:
            if y_true.shape == torch.Size([1]):
                continue
            x, y_true = x.cuda(), y_true.cuda()

            y_pred = model(x)   # Forward
            # print("yp:", y_pred)

            loss = criterion(y_pred, y_true)  # Calculate training loss
            print("  ", loss.item())
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
