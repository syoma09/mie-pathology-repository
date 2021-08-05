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
import numpy
import matplotlib.pyplot as plt
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile

from cnn.metrics import ConfusionMatrix


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)

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

    # Console write
    #print("    valid loss    : {:3.3}".format(metrics['loss']))
    #print("      accuracy    : {:3.3}".format(metrics['cmat'].accuracy()))
    #print("      f-measure   : {:3.3}".format(metrics['cmat'].f1inv))
    #print("      precision   : {:3.3}".format(metrics['cmat'].npv))
    #print("      specificity : {:3.3}".format(metrics['cmat'].specificity))
    #print("      recall      : {:3.3}".format(metrics['cmat'].tnr))
    #print("      Matrix:")
    #print(metrics['cmat'])

def create_model():

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

    return net

def main():
    src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    #root = Path("~/data/_out/mie-pathology/").expanduser()
    dataset_root = Path("~/root/mnt/cache").expanduser() / os.environ.get('USER'                                                                                                                                    ) / 'mie-pathology'
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
        PatchDataset(dataset_root, yml['train']), batch_size=batch_size, shuffle                                                                                                                                    =True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
     PatchDataset(dataset_root, yml['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    net = create_model().to(device)
    net.load_state_dict(torch.load(
    dataset_root / '20210715_152352model00177.pth' , map_location=device))
    fig ,ax = plt.subplots()
    d_today = datetime.date.today()
    with torch.no_grad():
        net.eval()
        y_true_plot = []
        y_pred_plot = []
        for batch,(x, y_true) in enumerate(valid_loader):
            x,y_true = x.to(device), y_true.to(device)
            y_pred = net(x)   # Forward
            print(y_pred.cpu().numpy().flatten().tolist())
            j = 0.
            k = 0.
            y_pred_num = 0.
            y_true_num = 0.
            for j in y_true.cpu().numpy().tolist():
                y_true_num += j
            for k in y_pred.cpu().numpy().flatten().tolist():
                y_pred_num += k
            y_true_plot.append(y_true_num / y_true.numel())
            y_pred_plot.append(y_pred_num / y_pred.numel())
        i = 0
        for i in range(len(y_true_plot)):
            print(y_true_plot)
            print(y_pred_plot)
            ax.plot(y_true_plot[i],y_pred_plot[i],'.')
            fig.savefig(f'valid{d_today}00177.png')
        print(f'valid00177.png')

if __name__ == '__main__':
    main()
