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
import AutoEncoder
import train_time
import random
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile

from cnn.metrics import ConfusionMatrix


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
            # torchvision.transforms.Normalize(0.5, 0.5)
        ])

        self.paths = []
        for subject in subjects:
            self.paths += list((root / subject).iterdir())
            '''print(subject)
            path = []
            path += list((root / subject).iterdir())
            if(subject == "57-10" or subject == "57-11"):
                self.paths += random.sample(path,4000)
            elif(subject == "38-4" or subject == "38-5"):
                self.paths += random.sample(path,len(path))
            elif(len(path) < 2000):
                self.paths += random.sample(path,len(path))
            else:
                self.paths+= random.sample(path,2000)'''
        #print(self.paths[0])
        print(len(self.paths))
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
        #label /= 90.
        # Classsification
        if(label < 12):
            label_class = 0
        elif(label < 24):
            label_class = 1
        elif(label < 36):
            label_class = 2
        elif(label < 48):
            label_class = 3
        # Tensor
        label = torch.tensor(label, dtype=torch.float)

        return img, label, label_class

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
    net = AutoEncoder.create_model()
    
    net.load_state_dict(torch.load(
        dataset_root / '20220222_032128model00257.pth', map_location=device)
    )
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    net.dec = nn.Sequential(
        #nn.Linear(512, 512, bias=True),
        #nn.Linear(512, 512, bias=True),
        nn.Linear(512, 4, bias=True),
        nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    '''for param in net.parameters():
        param.requires_grad = False
    last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True'''
    net.load_state_dict(torch.load(
        dataset_root / '20220317_114715model04000.pth', map_location=device)
        )
    net = net.to(device)
    # net = torch.nn.DataParallel(net).to(device)

    return net

def main():
    src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    #root = Path("~/data/_out/mie-pathology/").expanduser()
    dataset_root = Path('/mnt/cache')/os.environ.get('USER') / 'mie-pathology'
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
        PatchDataset(dataset_root, yml['train']), batch_size=batch_size, shuffle=True,num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
     PatchDataset(dataset_root, yml['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    net = AutoEncoder.create_model().to(device)
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    net.dec = nn.Sequential(
        #nn.Linear(512, 512, bias=True),
        #nn.Linear(512, 512, bias=True),
        nn.Linear(512, 4, bias=True),
        nn.Softmax(dim=1)
        # nn.Sigmoid()
     )
    '''for param in net.parameters():
        param.requires_grad = False
    last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
    param.requires_grad = True'''
    net.load_state_dict(torch.load(
    dataset_root / '20220317_114715model03888.pth', map_location=device))
    net = net.to(device)
    fig ,ax = plt.subplots()
    d_today = datetime.date.today()
    with torch.no_grad():
        net.eval()
        output_and_label = []
        y_true_plot = []
        valid_loss_tensor_plot = []
        j = 0
        for batch,(x, y_true, y_class) in enumerate(valid_loader):
            j += 1
            print(j)
            x,y_true,y_class = x.to(device), y_true.to(device), y_class.to(device)
            y_pred = net(x)   # Forward
            valid_loss_tensor = train_time.valid_loss(y_pred,y_true)
            #print(f'valid_loss_tensor : {valid_loss_tensor}')
            #output_and_label.append((valid_loss_tensor, y_true))
            for k  in range(len(y_pred)):
                y_true_plot.append(y_true[k].cpu().numpy().tolist())
                #print(y_true[k])
                #print('\n')
                valid_loss_tensor_plot.append(valid_loss_tensor[k].cpu().numpy().tolist())
            #print(f'valid_loss_tensor_plot : {valid_loss_tensor_plot}')
            #for k range(len(y_true):
            #print(f'y_true_plot : {y_true_plot}')
            #print(f'output_and_label : {output_and_label[0]}')
        output_and_label.append((valid_loss_tensor_plot, y_true_plot))
        #print(f'y_true_plot : {len(y_true_plot)}')
        #print(f'valid_loss_tensor_plot : {valid_loss_tensor_plot}')
        #print(f'output_and_label : {len(output_and_label)}')
        for i in range(len(valid_loss_tensor_plot)):
            print(f'i : {i}')
            #ax.plot(output_and_label[0],output_and_label[1],'.')
            ax.plot(valid_loss_tensor_plot[i],y_true_plot[i],'.')
            fig.savefig(f'valid{d_today}00000.png')
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        plt.show()
        print(f'valid00000.png')

if __name__ == '__main__':
    main()