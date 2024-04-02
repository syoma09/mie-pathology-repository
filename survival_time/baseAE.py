#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
from PIL import Image, ImageFile

from dataset_path import load_annotation

from aipatho.svs import TumorMasking
from data.dataset import create_dataset
from aipatho.model import AutoEncoder2
from aipatho.utils.directory import get_logdir, get_cache_dir


# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'
if torch.cuda.is_available():
    cudnn.benchmark = True


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []
        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
        ]

        # Random shuffle
        random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))
        self.__num_class = 3
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([l for _, l in self.__dataset if l <= 11]))
        print('  # of 1  :', len([l for _, l in self.__dataset if (11 < l) & (l <= 22)]))
        print('  # of 2  :', len([l for _, l in self.__dataset if (22 < l) & (l <= 33)]))
        print('  # of 3  :', len([l for _, l in self.__dataset if (33 < l) & (l <= 44)]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))

        '''self.paths = []
        for subject in subjects:
            print(subject)
            path = []
            path += list((root / subject).iterdir())
            if(subject == "57-10" or subject == "57-11"):
                self.paths += random.sample(path,4000)
            elif(subject == "38-4" or subject == "38-5"):
                self.paths += random.sample(path,len(path))
            elif(len(path) < 2000):
                self.paths += random.sample(path,len(path))
            else:
                self.paths+= random.sample(path,2000)
            self.paths += list((root / subject).iterdir())'''
        #print(self.paths[0])
        print(len(self.__dataset))
    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item: int) -> (torch.Tensor, torch.Tensor):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
    # img = self.data[item, :, :, :].view(3, 32, 32)
        path, label = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        #label /= 90.
        '''if(label < 13):
            label_class = 0
        elif(label < 34):
            label_class = 1
        elif(label < 67):
            label_class = 2
        if(label < 11):
            label_class = 0
        elif(label < 22):
            label_class = 1
        elif(label < 33):
            label_class = 2
        elif(label < 44):
            label_class = 3
        elif(label < 44):
            label_class = 4
        elif(label < 36):
            label_class = 5
        elif(label < 42):
            label_class = 6
        elif(label < 48):
            label_class = 7
        elif(label < 24):
            label_class = 11
        elif(label < 26):
            label_class = 12
        elif(label < 28):
            label_class = 13
        elif(label < 30):
            label_class = 14
        elif(label < 32):
            label_class = 15
        elif(label < 34):
            label_class = 16
        elif(label < 36):
            label_class = 17
        elif(label < 38):
            label_class = 18
        elif(label < 40):
            label_class = 19
        elif(label < 42):
            label_class = 20
        elif(label < 44):
            label_class = 21
        elif(label < 46):
            label_class = 22
        elif(label < 48):
            label_class = 23
        elif(label < 50):
            label_class = 24
        elif(label < 52):
            label_class = 25
        elif(label < 54):
            label_class = 26
        elif(label < 56):
            label_class = 27
        elif(label < 58):
            label_class = 28
        elif(label < 60):
            label_class = 29
        elif(label < 62):
            label_class = 30
        elif(label < 64):
            label_class = 31
        elif(label < 66):
            label_class = 32
        elif(label < 68):
            label_class = 33'''
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

    # Build data loader
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
