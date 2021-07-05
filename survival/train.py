#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import re
import random
from pathlib import Path
from joblib import Parallel, delayed

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image, ImageOps

from cnn.metrics import ConfusionMatrix
from data.svs import save_patches
from survival import get_dataset_root_path, PatchDataset, create_model


# Set CUDA device
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

# Check RAM capacity
with open('/proc/meminfo', 'r') as f:
    mem_total_str = [line for line in f.readlines() if line.startswith('MemTotal')]
    mem_total = re.findall(r'[0-9]+', mem_total_str[0])[0]
    mem_total = int(mem_total) / 1024 / 1024  # kB -> mB -> gB
    mem_total -= 4  # Keep 4GB for system
    print(mem_total)


def create_dataset(src: Path, dst: Path, annotation: Path):
    # Load annotation
    df = pd.read_csv(annotation)
    print(df)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']

        path_svs = src / "svs" / f"{number}.svs"
        path_xml = src / "xml" / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue

        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)

        base = subject_dir / 'patch'
        size = 512, 512
        stride = size
        resize = 256, 256
        args.append((path_svs, path_xml, base, size, stride, resize))

        # # Serial execution
        # save_patches(path_svs, path_xml, base, size=size, stride=stride)

    # Approx., 1 thread use 20GB
    # n_jobs = int(mem_total / 20)
    n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize)
        for path_svs, path_xml, base, size, stride, resize in args
    ])


def main():
    # target = '3os'
    target = '2dfs'
    # dataset_root = Path("~/data/_out/mie-pathology/").expanduser()
    dataset_root = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology' / f"survival_{target}"

    annotation_path = Path(
        f"~/workspace/mie-pathology/_data/survival_{target}.csv"
    ).expanduser()

    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
        create_dataset(
            src=Path("~/workspace/mie-pathology/_data/").expanduser(),
            dst=dataset_root,
            annotation=annotation_path
        )

    # return
    epochs = 1000
    batch_size = 128     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT

    # Load annotations
    annotation = {
        'train': [], 'valid': [], 'test': [], 'IGNORE': []
    }
    for _, row in pd.read_csv(annotation_path).iterrows():
        annotation[
            # Switch train/valid by tvt-column value (0: train, 1: valid)
            ['train', 'valid', 'test', 'IGNORE'][int(row['tvt'])]
        ].append((row['number'], row['label']))     # Append annotation tuple

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )

    '''
    モデルの構築
    '''
    model = create_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()              # Need Sigmoid
    # criterion = nn.BCELoss(reduction='sum')              # Need Sigmoid
    criterion = nn.BCEWithLogitsLoss()

    tensorboard = SummaryWriter(log_dir='./logs')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")

        # Initialize metric values on epoch
        metrics = {
            'train': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }, 'valid': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }
        }

        # Switch to training mode
        model.train()

        for batch, (x, y_true) in enumerate(train_loader):
            optimizer.zero_grad()

            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)   # Forward
            y_pred = y_pred.logits  # To convert InceptionOutputs -> Tensor
            # y_pred = torch.sigmoid(y_pred)
            # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            # print(y_true)
            # print(y_pred)

            loss = criterion(y_pred, y_true)  # Calculate training loss
            loss.backward()     # Backward propagation
            optimizer.step()    # Update parameters

            # Logging
            metrics['train']['loss'] += loss.item() / len(train_loader)
            metrics['train']['cmat'] += ConfusionMatrix(y_pred.cpu(), y_true.cpu())
            # Screen output
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")

        print('')
        print('  Saving model...')
        torch.save(model.state_dict(), dataset_root / f"{model_name}{epoch:05}.pth")

        # Switch to evaluation mode
        model.eval()

        # Calculate validation metrics
        with torch.no_grad():
            for i, (x, y_true) in enumerate(valid_loader):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)  # Prediction
                # y_pred = torch.sigmoid(y_pred)
                # y_pred = torch.nn.functional.softmax(y_pred, dim=1)

                loss = criterion(y_pred, y_true)  # Calculate validation loss
                # print(loss.item())
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)

                print("\r  Validating... ({:6}/{:6})[{}]".format(
                    i, len(valid_loader),
                    ('=' * (30 * i // len(valid_loader)) + " " * 30)[:30]
                ), end="")

        # Console write
        print("")
        print("    train loss      : {:3.3}".format(metrics['train']['loss']))
        print("          precision : {:3.3}".format(metrics['train']['cmat'].precision()))
        print("          recall    : {:3.3}".format(metrics['train']['cmat'].recall()))
        print("          accuracy  : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        print("          f-measure : {:3.3}".format(metrics['train']['cmat'].f1()))
        print(metrics['train']['cmat'])
        print("    valid loss      : {:3.3}".format(metrics['valid']['loss']))
        print("          precision : {:3.3}".format(metrics['valid']['cmat'].precision()))
        print("          recall    : {:3.3}".format(metrics['valid']['cmat'].recall()))
        print("          accuracy  : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        print("          f-measure : {:3.3}".format(metrics['valid']['cmat'].f1()))
        print("        Matrix:")
        print(metrics['valid']['cmat'])
        # Write tensorboard
        tensorboard.add_scalar('train_loss', metrics['train']['loss'], epoch)
        tensorboard.add_scalar('train_f1', metrics['train']['cmat'].f1(), epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_prec', metrics['valid']['cmat'].precision(), epoch)
        tensorboard.add_scalar('valid_rec', metrics['valid']['cmat'].recall(), epoch)
        tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)


if __name__ == '__main__':
    main()
