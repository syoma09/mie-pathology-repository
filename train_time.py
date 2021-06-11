#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile

from cnn.metrics import ConfusionMatrix

# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, subjects):
        super(PatchDataset, self).__init__()

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0.5, 0.5)
        ])

        self.paths = []
        for subject in subjects:
            self.paths += list((root / subject).iterdir())

        # print(self.paths[0])
        # print(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """

        # img = self.data[item, :, :, :].view(3, 32, 32)
        img = Image.open(self.paths[item]).convert('RGB')
        img = self.transform(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        name = self.paths[item].name                # Filename
        label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        label /= 90.
        # Tensor
        label = torch.tensor(label, dtype=torch.float)

        return img, label

    # @classmethod
    # def load_list(cls, root):
    #     # 顎骨正常データ取得と整形
    #
    #     with open(root, "rb") as f:
    #         output = pickle.load(f)
    #
    #     return output
    #
    # @classmethod
    # def load_torch(cls, _list):
    #     output = torch.cat([_dict["data"].view(1, 3, 32, 32) for _dict in _list],
    #                        dim=0)
    #
    #     return output
    #
    # @classmethod
    # def load_necrosis(cls, root):
    #     data = cls.load_list(root)
    #     data = cls.load_torch(data)
    #
    #     return data

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
#
#         return torch.abs(c_pred - c_true)


def main():
    src = Path("~/workspace/mie-pathology/_data/").expanduser()
    root = Path("~/data/_out/mie-pathology/").expanduser()

    epochs = 10000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT

    # Load train/valid yaml
    with open(src / "survival_time.yml", "r") as f:
        yml = yaml.safe_load(f)

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(root, yml['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(root, yml['valid']), batch_size=batch_size,
        num_workers=num_workers
    )

    '''
    モデルの構築
    '''
    # model = torchvision.models.resnet152(pretrained=False)
    # model = torchvision.models.resnet152(pretrained=True)
    net = torchvision.models.resnet50(pretrained=True)
    # print(model)

    # Replace FC layer
    num_features = net.fc.in_features
    # print(num_features)  # 512
    net.fc = nn.Sequential(
        nn.Linear(num_features, 1, bias=True),
        # nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    net = net.to(device)
    # net = torch.nn.DataParallel(net).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # criterion = WeightedProbLoss(classes=4)
    criterion = nn.MSELoss()

    tensorboard = SummaryWriter(log_dir='./logs')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")

        # Switch to training mode
        net.train()

        train_loss = 0.
        for batch, (x, y_true) in enumerate(train_loader):
            optimizer.zero_grad()

            x, y_true = x.to(device), y_true.to(device)
            y_pred = net(x)   # Forward
            # print("yp:", y_pred)

            loss = criterion(y_pred, y_true)  # Calculate training loss
            loss.backward()     # Backward propagation
            optimizer.step()    # Update parameters

            # Logging
            train_loss += loss.item() / len(train_loader)
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
            tensorboard.add_scalar(
                'train_loss', loss.item(), epoch * batch_size + batch
            )
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), root / f"{model_name}{epoch:05}.pth")

        # Switch to evaluation mode
        net.eval()
        # On training data

        # Initialize validation metric values
        metrics = {
            'train': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }, 'valid': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }
        }
        # Calculate validation metrics
        with torch.no_grad():
            for x, y_true in valid_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = net(x)  # Prediction

                loss = criterion(y_pred, y_true)  # Calculate validation loss
                # print(loss.item())
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)

            for x, y_true in train_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = net(x)  # Prediction

                metrics['train']['loss'] += criterion(y_pred, y_true).item() / len(train_loader)
                # metrics['train']['cmat'] += ConfusionMatrix(y_pred, y_true)

        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        # print("    valid loss: {:3.3}".format(metrics['valid']['loss']))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)


if __name__ == '__main__':
    main()
