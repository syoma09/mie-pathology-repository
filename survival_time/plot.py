#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path
import datetime
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import AutoEncoder
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.stats import norm


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, subjects):
        super(PatchDataset, self).__init__()

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
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

        # Classsification
        '''if(label < 10):
            label = 0
        elif(label < 31):
            label = 1
        elif(label < 67):
            label = 2'''
        # Tensor
        label = torch.tensor(label, dtype=torch.float)

        return img, label

    # Console write
    #print("    valid loss    : {:3.3}".format(metrics['loss']))
    #print("      accuracy    : {:3.3}".format(metrics['cmat'].accuracy()))
    #print("      f-measure   : {:3.3}".format(metrics['cmat'].f1inv))
    #print("      precision   : {:3.3}".format(metrics['cmat'].npv))
    #print("      specificity : {:3.3}".format(metrics['cmat'].specificity))
    #print("      recall      : {:3.3}".format(metrics['cmat'].tnr))
    #print("      Matrix:")
    #print(metrics['cmat'])
class AutoEncoder2(torch.nn.Module):
     def __init__(self, enc, dec):
         super().__init__()
         self.enc = enc
         self.dec = dec
     def forward(self, x):
         x = self.enc(x)
         x = self.dec(x)
         return x

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def main():
    src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    #root = Path("~/data/_out/mie-pathology/").expanduser()
    dataset_root = Path('/mnt/cache') / os.environ.get('USER') / 'mie-pathology'
    batch_size = 32
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True,exist_ok=True)
    # # Subject
    # result = {}
    # for name, cls in annotation['test']:
    #     cmat = evaluate(
    #         dataset_root=get_dataset_root_path(target=target),
    #         subjects=[(name, cls)]
    #     )
    #
    #     result[name] = {
    #         "true": cls,
    #         "pred": np.argmin([cmat.fp + cmat.tn, cmat.fp + cmat.fn]),
    #         "rate": cmat.accuracy()
    #     }
    # print(result)
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
     PatchDataset(dataset_root, yml['train']), batch_size=batch_size,
        num_workers=num_workers
    )
    '''valid_loader = torch.utils.data.DataLoader(
     PatchDataset(dataset_root, yml['valid']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )'''
    net = AutoEncoder.create_model().to(device)
    net.load_state_dict(torch.load(
    dataset_root/ '20220321_165400model00269.pth', map_location=device))
    output_and_label = []
    #print(net)
    for batch, (x, y_true) in enumerate(train_loader):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = net(x) 
            print("\r  Batch({:6}/{:6})[{}]: ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            ), end="")
    output_and_label.append((y_pred, x))
    org, img = output_and_label[-1]
    img = img[0:1]
    org = org[0:1]
    plt.figure(figsize=(1, 1))
    imshow(org.cpu())
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig('figure1.png')
    plt.figure(figsize=(1, 1))
    imshow(img.cpu())
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig('figure2.png') # -----(2)
    '''for batch, (x, y_true) in enumerate(valid_loader):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = net(x)
        print("\r  Batch({:6}/{:6})[{}]: ".format(
                batch, len(valid_loader),
                 ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30]
             ), end="")
    output_and_label.append((y_pred, x))
    org, img = output_and_label[-1]
    img = img[0:3]
    org = org[0:3]

    plt.figure(figsize=(3, 3))
    imshow(org.cpu())
    plt.savefig('figure7.png')
    plt.figure(figsize=(3, 3))
    imshow(img.cpu())
    plt.savefig('figure8.png') # -----(2)'''

if __name__ == '__main__':
     main()
