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

from aipatho.metrics import ConfusionMatrix
from aipatho.svs import save_patches
from survival import load_annotation, get_dataset_root_path, PatchDataset, create_model


# Set CUDA device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

# Check RAM capacity
with open('/proc/meminfo', 'r') as f:
    mem_total_str = [line for line in f.readlines() if line.startswith('MemTotal')]
    mem_total = re.findall(r'[0-9]+', mem_total_str[0])[0]
    mem_total = int(mem_total) / 1024 / 1024  # kB -> mB -> gB
    mem_total -= 4  # Keep 4GB for system
    print(mem_total)


def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        index: int = None, region: int = None
):
    # Load annotation
    df = pd.read_csv(annotation)
    print(df)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']

        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Subject #{number} already exists. Skip.")
            continue

        path_svs = src / f"{number}.svs"
        path_xml = src / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue

        base = subject_dir / 'patch'
        resize = 256, 256
        args.append((path_svs, path_xml, base, size, stride, resize))

        # # Serial execution
        # save_patches(path_svs, path_xml, base, size=size, stride=stride)

    # Approx., 1 thread use 20GB
    n_jobs = int(mem_total / 20)
    # n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize, index, region)
        for path_svs, path_xml, base, size, stride, resize in args
    ])


def main():
    patch_size = 1024, 1024
    stride = 512, 512
    # patch_size = 256, 256

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)

    annotation_path = Path(
        # "../_data/20220610_3os.csv"
        "../_data/20220610_3os/cv0.csv"
        # "../_data/20220610_3os/cv1.csv"
        # "../_data/20220610_3os/cv2.csv"
        # "../_data/20220610_3os/cv3.csv"
    ).expanduser()

    # Create dataset if not exists
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride
    )
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)

    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=1, region=None
    )

    # Load annotations
    annotation = load_annotation(annotation_path)
    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=16, shuffle=True,
        num_workers=os.cpu_count() // 2
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']),
        batch_size=train_loader.batch_size*2,
        num_workers=os.cpu_count() // 2
    )

    '''
    Build model
    '''
    model = create_model().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()              # Need Sigmoid
    # criterion = nn.BCELoss(reduction='sum')              # Need Sigmoid
    criterion = nn.BCEWithLogitsLoss()

    tensorboard = SummaryWriter(log_dir=str(log_root))

    epochs = 300
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
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)   # Forward
            # y_pred = y_pred.logits  # To convert InceptionOutputs -> Tensor

            loss = criterion(y_pred, y_true)  # Calculate training loss
            optimizer.zero_grad()
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
        torch.save(model.state_dict(), log_root / f"model{epoch:05}.pth")

        # Switch to evaluation mode
        model.eval()

        # Calculate validation metrics
        with torch.no_grad():
            for i, (x, y_true) in enumerate(valid_loader):
                x, y_true = x.to(device), y_true.to(device)
                y_pred = model(x)  # Prediction

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
        print("    train loss  : {:3.3}".format(metrics['train']['loss']))
        print("          f1inv : {:3.3}".format(metrics['train']['cmat'].f1inv))
        print("          npv   : {:3.3}".format(metrics['train']['cmat'].npv))
        print("          tnr   : {:3.3}".format(metrics['train']['cmat'].tnr))
        print(metrics['train']['cmat'])
        print("    valid loss  : {:3.3}".format(metrics['valid']['loss']))
        print("          f1inv : {:3.3} (f1={:3.3})".format(
            metrics['valid']['cmat'].f1inv,
            metrics['valid']['cmat'].f1
        ))
        print("          npv   : {:3.3}".format(metrics['valid']['cmat'].npv))
        print("          tnr   : {:3.3}".format(metrics['valid']['cmat'].tnr))
        print("        Matrix:")
        print(metrics['valid']['cmat'])
        # Write tensorboard
        for tv in ['train', 'valid']:
            # Loss
            tensorboard.add_scalar(f"{tv}_loss", metrics[tv]['loss'], epoch)
            # For ConfusionMatrix
            for m_name in ['f1', "f1inv", "npv", "tpr", "precision", "recall", "tn", "tp", "fn", "fp"]:
                tensorboard.add_scalar(f"{tv}_{m_name}", getattr(metrics[tv]['cmat'], m_name), epoch)


if __name__ == '__main__':
    main()
