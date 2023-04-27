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
import matplotlib.pyplot as plt
import AutoEncoder
import train_time
import VAE
import random
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path

from cnn.metrics import ConfusionMatrix


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations,flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.__dataset = []
        self.paths = []
        for subject in annotations:
            self.paths += [
                path  # Same label for one subject
                for path in (root / subject).iterdir()
            ]
        if (flag == 0):
            self.__dataset += random.sample(self.paths,len(self.paths))
            #self.__dataset += random.sample(self.paths,1000)
        else:
            self.__dataset += random.sample(self.paths,flag)
        #self.__dataset += random.sample(self.paths,len(self.paths))

        # Random shuffle
        random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))
        
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
    
    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
        # img = self.data[item, :, :, :].view(3, 32, 32)
        path = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)
        true_class = 0.
        if("not" in str(path)):
            true_class = 1
        else:
            true_class = 0
        """self.__num_class = 2
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([if(class == 0)]))
        print('  # of 1  :', len([if(class == 1)]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))"""
        # Tensor
        true_class = torch.tensor(true_class, dtype=torch.float)
        return img, true_class


def main():
    #patch_size = 512,512
    patch_size = 256,256
    stride = 256,256
    #stride = 128,128
    index = 2
    batch_size = 32
    # patch_size = 256, 256
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )
    
    dataset_root_not = get_dataset_root_not_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )
    
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    
    annotation_path = Path(
        "../_data/survival_time_cls/20221206_Auto.csv"
        #"../_data/survival_time_cls/20220725_aut1.csv"
    ).expanduser()
    annotation = load_annotation(annotation_path)
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

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    train_dataset = []
    valid_dataset = []
    flag = 0
    train_dataset.append(PatchDataset(dataset_root, annotation['train'],flag))
    flag = len(train_dataset[0])
    train_dataset.append(PatchDataset(dataset_root_not, annotation['train'],flag))
    flag = 0
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid'], flag))
    flag = len(valid_dataset[0])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset),batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_dataset.append(PatchDataset(
        dataset_root_not, annotation['valid'], flag))
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    print("valid_loader:", len(valid_loader))
    # model = torchvision.models.resnet152(pretrained=False)
    # model = torchvision.models.resnet152(pretrained=True)
    #z_dim = 512
    z_dim = 512
    #net = AutoEncoder.create_model()
    net = VAE.VAE(z_dim)
    net.load_state_dict(torch.load(
        #dataset_root / '20220222_032128model00257.pth', map_location=device)
        road_root / '20230426_153425' / 'model00048.pth', map_location=device)
    )
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    """net.dec = nn.Sequential(
        #nn.Linear(512, 512, bias=True),
        #nn.Linear(512, 512, bias=True),
        nn.Linear(512, 4, bias=True),
        nn.Softmax(dim=1)
        # nn.Sigmoid()
    )"""
    '''for param in net.parameters():
        param.requires_grad = False
    last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True'''
    net = net.to(device)
    # net = torch.nn.DataParallel(net).to(device)
    
    """with torch.no_grad():
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
        print(f'valid00000.png')"""
    cm = plt.get_cmap("tab10") # カラーマップの用意
    with torch.no_grad():
        fig_plot, ax_plot = plt.subplots(figsize=(9, 9))
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 9))
        for batch,(input, labels) in enumerate(train_loader):
            print("\r  Printing... ({:6}/{:6})[{}]: ".format(
                    batch, len(train_loader),
                    ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                ), end="")
            input, labels = input.to(device),labels.to(device)
            # 学習済みVAEに入力を与えたときの潜在変数を抽出
            _, z, _ , _= net(input)
            z = z.cpu().detach().numpy()
            # 各クラスごとに可視化する
            for k in range(2):
                cluster_indexes = np.where(labels.cpu().detach().numpy() == k)[0]
                ax_plot.plot(z[cluster_indexes,0], z[cluster_indexes,1], "o", ms=4, color=cm(k))
        fig_plot.legend(["High-grade area","Other tumor area"])
        fig_plot.savefig(f"./train_space_z_{z_dim}_{batch}_plot.png")
        for batch,(input, labels) in enumerate(valid_loader):
            print("\r  Printing... ({:6}/{:6})[{}]: ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                ), end="")
            input, labels = input.to(device),labels.to(device)
            # 学習済みVAEに入力を与えたときの潜在変数を抽出
            _, z, _ , _= net(input)
            z = z.cpu().detach().numpy()
            # 各クラスごとに可視化する
            for k in range(2):
                cluster_indexes = np.where(labels.cpu().detach().numpy() == k)[0]
                ax_plot.plot(z[cluster_indexes,0], z[cluster_indexes,1], "o", ms=4, color=cm(k))
        fig_plot.legend(["High-grade area","Other tumor area"])
        
        fig_plot.savefig(f"./valid_space_z_{z_dim}_{batch}_plot.png")
        #fig_scatter.savefig(f"./latent_space_z_{z_dim}_{batch}_scatter.png")
        plt.close(fig_plot)
        plt.close(fig_scatter)
            

if __name__ == '__main__':
    main()
