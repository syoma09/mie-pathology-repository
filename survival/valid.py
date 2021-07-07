#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from cnn.metrics import ConfusionMatrix
from survival import load_annotation, PatchDataset, create_model


# Set CUDA device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True


def main():
    # target = '3os'
    target = '2dfs'
    dataset = "valid"
    epochs = 1000

    dataset_root = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology' / f"survival_{target}"

    # Load annotations
    annotation = load_annotation(Path(
        f"~/workspace/mie-pathology/_data/survival_{target}.csv"
    ).expanduser())
    # Init DataLoader
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation[dataset]),
        batch_size=1024, num_workers=os.cpu_count() // 2
    )

    '''
    モデルの構築
    '''
    model = create_model().to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    tensorboard = SummaryWriter(log_dir='./logs')

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")

        path = dataset_root / f"20210702_175146model{epoch:05}.pth"
        if not path.exists():
            break

        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        # Calculate validation metrics
        loss = 0
        cmat = ConfusionMatrix(None, None)
        with torch.no_grad():
            for i, (x, y_true) in enumerate(valid_loader):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)

                loss += criterion(y_pred, y_true).item() / len(valid_loader)
                cmat += ConfusionMatrix(y_pred, y_true)

                print("\r  Validating... ({:6}/{:6})[{}]".format(
                    i, len(valid_loader),
                    ('=' * (30 * i // len(valid_loader)) + " " * 30)[:30]
                ), end="")

        # Console write
        print("")
        print("    valid loss      : {:3.3}".format(loss))
        print("          f-measure : {:3.3}".format(cmat.f1inv))
        print("        Matrix:")
        print(cmat)
        # Write tensorboard
        tensorboard.add_scalar('valid_loss', loss, epoch)
        tensorboard.add_scalar('valid_f1inv', cmat.f1inv, epoch)
        tensorboard.add_scalar('valid_npv', cmat.npv, epoch)
        tensorboard.add_scalar('valid_tnr', cmat.tnr, epoch)


if __name__ == '__main__':
    main()
