#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn

from cnn.metrics import ConfusionMatrix
from survival import load_annotation, get_dataset_root_path, PatchDataset, create_model


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)


def main():
    target = '2dfs'

    annotation = load_annotation(Path(
        f"~/workspace/mie-pathology/_data/survival_{target}.csv"
    ).expanduser())

    # データ読み込み
    dataset_root = get_dataset_root_path(target=target)
    data_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']),
        # PatchDataset(dataset_root, annotation['test']),
        batch_size=512, shuffle=True, num_workers=os.cpu_count() // 2
    )

    '''
    モデルの構築
    '''
    model = create_model().to(device)
    model.load_state_dict(torch.load(
        dataset_root / "20210702_175146model00186.pth",
        map_location=device
    ))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # Initialize validation metric values
    metrics = {
        'loss': 0.,
        'cmat': ConfusionMatrix(None, None)
    }
    # Calculate validation metrics
    with torch.no_grad():
        for i, (x, y_true) in enumerate(data_loader):
            print("\r Test ({:5}/{:5})".format(
                i, len(data_loader)
            ), end='')

            y_true = y_true.to(device)
            y_pred = model(x.to(device))    # Prediction

            loss = criterion(y_pred, y_true)  # Calculate validation loss
            # print(loss.item())
            metrics['loss'] += loss.item() / len(data_loader)
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
