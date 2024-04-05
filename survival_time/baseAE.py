#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision

from aipatho.svs import TumorMasking
from aipatho.model import AutoEncoder2
from aipatho.utils.directory import get_logdir, get_cache_dir
from aipatho.dataset import PatchDataset, TimeToTime, load_annotation, create_dataset

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'
if torch.cuda.is_available():
    cudnn.benchmark = True


# class WeightedProbLoss(nn.Module):
#     def __init__(self, classes):
#         super(WeightedProbLoss, self).__init__()
#
#         if isinstance(classes, int):
#             classes = [i for i in range(classes)]
#
#         self.classes = torch.Tensor(classes).to(device)
#
#     def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#         """
#
#         :param pred:    Probabilities of each class
#         :param true:    1-hot vector
#         :return:
#         """
#
#         c_pred = torch.sum(torch.mul(pred, self.classes))
#         c_true = torch.argmax(true)





def main():
    patch_size = 512, 512
    # patch_size = 256, 256
    stride = 512, 512
    target = TumorMasking.FULL

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=target
    )
    print(dataset_root)

    annotation_path = Path(
        "_data/survival_time_cls/20220413_aut2.csv"
    )
    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=1, region=None,
        target=target
    )
    # Load annotations
    annotation = load_annotation(annotation_path)

    # Log, epoch-model output directory
    epochs = 10_000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    # # Load train/valid yaml
    # with open(src / "survival_time.yml", "r") as f:
    #     yml = yaml.safe_load(f)

    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Build data loader
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], transform=transform, labeler=TimeToTime()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], transform=transform, labeler=TimeToTime()),
        batch_size=batch_size,
        num_workers=num_workers
    )

    '''
    モデルの構築
    '''
    net = AutoEncoder2().to(device)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()

    logdir = get_logdir()
    tensorboard = SummaryWriter(log_dir=str(logdir))

    print(net)
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()

        train_loss = 0.
        for batch, (x, _) in enumerate(train_loader):
            # Init optimizer
            optimizer.zero_grad()

            # Calc loss
            x = x.to(device)
            y_pred = net(x)   # Forward
            loss = criterion(y_pred, x)

            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters

            # Logging
            train_loss += loss.item() / len(train_loader)
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
        print("train_loss", train_loss)
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), logdir / f"state{epoch:05}.pth")

        # Switch to evaluation mode
        net.eval()

        # Calculate validation metrics
        valid_loss = 0.
        with torch.no_grad():
            # valid_loss = 0.
            for batch, (x, _) in enumerate(valid_loader):
                x = x.to(device)
                y_pred = net(x)  # Prediction
                loss = criterion(y_pred, x)
                # Logging
                valid_loss += loss.item() / len(valid_loader)

        # Console write
        print("    valid loss: {:3.3}".format(valid_loss))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # Write tensorboard
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', valid_loss, epoch)


if __name__ == '__main__':
    main()
