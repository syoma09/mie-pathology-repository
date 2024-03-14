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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from contrastive_learning import Hparams,SimCLR_pl,AddProjection
from cnn.metrics import ConfusionMatrix
from AutoEncoder import create_model
from sklearn.manifold import TSNE

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.__dataset = []
        paths = []

        for subject in annotations:
            paths += [
                (path)  # Same label for one subject
                for path in (root / subject).iterdir()
            ]
        self.__dataset += random.sample(paths,len(paths) // 100)
        """if (flag == 0):
            self.__dataset += random.sample(self.paths,len(self.paths))
            #self.__dataset += random.sample(self.paths,1000)
        else:
            self.__dataset += random.sample(self.paths,flag)
        #self.__dataset += random.sample(self.paths,len(self.paths))"""
        

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
        #"../_data/survival_time_cls/20221206_Auto.csv"
        #"../_data/survival_time_cls/20220725_aut1.csv"
        "../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        #"../_data/survival_time_cls/fake_data.csv"
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
    #flag = 0
    train_dataset.append(PatchDataset(dataset_root, annotation['train']))
    #flag = len(train_dataset[0])
    train_dataset.append(PatchDataset(dataset_root_not, annotation['train']))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset),batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    #flag = 0
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid']))
    #flag = len(valid_dataset[0])
    valid_dataset.append(PatchDataset(dataset_root_not, annotation['valid']))
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    print("valid_loader:", len(valid_loader))
    # model = torchvision.models.resnet152(pretrained=False)
    # model = torchvision.models.resnet152(pretrained=True)
    
    net = create_model()
    net.load_state_dict(torch.load(
        #road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
        road_root / "20230928_160620" /'20230928_160625model00166.pth', map_location=device) #AE                               
    )
    
    net.dec = nn.Linear(512,128)
    net = net.to(device)
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
            y_pred = net(input)
            # 学習済みVAEに入力を与えたときの潜在変数を抽出
            clusters = KMeans(n_clusters = 4).fit(y_pred.cpu().detach().numpy())

            points = TSNE(n_components=2,perplexity = 5,random_state=0).fit_transform(y_pred.cpu().detach().numpy())            
            
            #tmp = np.stack(points,clusters.labels_,1)
            for i in range(len(y_pred)):
                for j in range(4):
                    if(clusters.labels_[i] == j):
                        ax_plot.scatter(points[0], points[1])
            
            
            """# 各クラスごとに可視化する
            for k in range(2):
                cluster_indexes = np.where(labels.cpu().detach().numpy() == k)[0]
                
                ax_plot.plot([cluster_indexes,0], z[cluster_indexes,1], "o", ms=4, color=cm(k))
                """
        #fig_plot.legend(["Survival","Non-Survival"])
        #fig_plot.legend(["Maligunant","Non-Maligunant"])
        fig_plot.savefig(f"train_kmeans_plot.png")
        """for batch,(input, labels) in enumerate(valid_loader):
            print("\r  Printing... ({:6}/{:6})[{}]: ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                ), end="")
            input, labels = input.to(device),labels.to(device)
            # 学習済みVAEに入力を与えたときの潜在変数を抽出
            z, _ = net(input)
            z = z.cpu().detach().numpy()
            # 各クラスごとに可視化する
            or k in range(2):
                cluster_indexes = np.where(labels.cpu().detach().numpy() == k)[0]
                ax_plot.plot(z[cluster_indexes,0], z[cluster_indexes,1], "o", ms=4, color=cm(k))
                
        #fig_plot.legend(["Survival","Non-Survival"])
        fig_plot.legend(["High-grade area","Other tumor area"])
        fig_plot.savefig(f"./valid_kmeans_plot.png")
        #fig_scatter.savefig(f"./latent_space_z_{z_dim}_{batch}_scatter.png")
        plt.close(fig_plot)
        plt.close(fig_scatter)"""
            

if __name__ == '__main__':
    main()