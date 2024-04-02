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
import random
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from scipy.stats import norm
from dataset_path import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from contrastive_learning import Hparams,SimCLR_pl,AddProjection
from create_soft_labels import estimate_value

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                                std=[0.229, 0.224, 0.225])
        ])
        self.__dataset = []
        data = []
        for subject,label in annotations:
            data += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
        ]
        self.__dataset = random.sample(data,len(data) )

        # Random shuffle
        random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
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

    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
        # img = self.data[item, :, :, :].view(3, 32, 32)
        path,label = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time
        if(label < 11):
            label_class = 0
        elif(label < 22):
            label_class = 1
        elif(label < 33):
            label_class = 2
        elif(label < 44):
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

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#前処理で行ったnormalizeを元に戻す関数
def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img 

def main():
    patch_size = 256, 256
    stride = 256, 256
    # patch_size = 256, 256
    index = 2
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
    
    # Log, epoch-model output directory
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        #"../_data/survival_time_cls/fake_data.csv"
        "../_data/survival_time_cls/20220726_cls.csv"
        #"../_data/survival_time_cls/20220726_cls/cv2.csv"
        #"../_data/survival_time_cls/fake_data.csv"
        #"../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        
    ).expanduser()
    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    # Existing subjects are ignored in the function
    """create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=None, region=None
    )"""
    # Load annotations
    annotation = load_annotation(annotation_path)
    #print(annotation)
    # echo $HOME == ~
    #src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    # Write dataset on SSD (/mnt/cache/)
    #dataset_root = Path("/mnt/cache").expanduser()/ os.environ.get('USER') / 'mie-pathology'
    '''if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)'''
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    """
    train_dataset = []
    valid_dataset = []
    train_dataset.append(PatchDataset(dataset_root, annotation['train']))
    train_dataset.append(PatchDataset(
        dataset_root_not, annotation['train']))
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid']))
    valid_dataset.append(PatchDataset(
        dataset_root_not, annotation['valid']))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    """
    #dataset_root = Path("/mnt/cache").expanduser()/ os.environ.get('USER') / 'mie-pathology'
    z_dim = 512
    train_config = Hparams()
    net = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512)
    #net = torchvision.models.resnet18()
    net.model.projection = nn.Sequential(
    #net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )
    net.load_state_dict(torch.load(
        
        #road_root / "20230824_155532" /'model00329.pth', map_location=device) #CL-Soft
        #road_root / "20230824_155636" /'model00976.pth', map_location=device) #CL-Hard
        road_root / "20240306_160314" /'model00117.pth', map_location=device) #CL-Soft
    )

    """net.load_state_dict(torch.load(
        #dataset_root / '20220222_032128model00257.pth', map_location=device)
        road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
    )"""
    """
    net = torchvision.models.resnet18()
    #net.model.projection = nn.Sequential(
    net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )

    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        #road_root / "20230530_205222" /'20230530_205224model00020.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #road_root / "20230712_131831" /'20230712_131837model00140.pth', map_location=device)
        #road_root / "20230720_170104" /'model00964.pth', map_location=device)
        #road_root / "20230727_180854" /'model00154.pth', map_location=device)
        #road_root / "20230824_155532" /'model00329.pth', map_location=device) 
        #road_root / "20230926_153532" /'model00083.pth', map_location=device)
        #road_root / "20230926_153622" /'model00685.pth', map_location=device) 
        #road_root / "20231013_191738" /'model00375.pth', map_location=device) 
        road_root / "20231204_163803" /'model00375.pth', map_location=device) #cv2
        #road_root / "20231204_163814" /'model00006.pth', map_location=device) #cv1?
        #road_root / "20231204_164043" /'model00352.pth', map_location=device)  #cv0?
    )"""

    output_and_label = []
    net = net.to(device)
    net.eval()
    print(net)
    fig1 ,ax1 = plt.subplots(figsize=(10, 10))
    with torch.no_grad():
        """for batch, (x,y_true,y_class) in enumerate(train_loader):
            x, y_true  = x.to(device), y_true.to(device)
            y_pred = net(x)
            #m = nn.Sigmoid()
            #y_pred = m(y_pred)
            y_pred = estimate_value(y_pred)
            y_pred = np.squeeze(y_pred)
            #print(y_pred.T)
            print("\r  Batch({:6}/{:6})[{}]:".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            ), end="")
            #y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            for i in range(len(y_pred)):
                if(y_class[i] == 0):
                    ax1.hist(y_pred, bins=5,  color='red')
                elif(y_class[i] == 1):
                    ax1.hist(y_pred, bins=5,  color='blue')
                elif(y_class[i] == 2):
                    ax1.hist(y_pred, bins=5,  color='green')
                elif(y_class[i] == 3):
                    ax1.hist(y_pred, bins=5,  color='yellow')
                #ax1.set_xlabel("Pred")
                #ax1.set_ylabel("OS")
                ax1.set_xlabel("Pred")
                ax1.set_ylabel("Freq")
                #ax1.set_xticks(np.arange(0,51,5))
                #ax1.set_yticks(np.arange(0,51,5))
        fig1.savefig(f'train_{batch}_pred_hist.png')
        plt.clf()
        plt.close()"""
        
        #fig3 = plt.figure(figsize=(8,6))
        #fig2, ax2 = plt.subplots(2, 2, tight_layout=True)
        #ax2 = fig2.add_subplot(1,1,1)
        x1,x2,x3,x4 = [],[],[],[]
        for batch, (x, y_true,y_class) in enumerate(valid_loader):
        #for batch, (x, y_true,y_class) in enumerate(train_loader):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = net(x)
            y_pred = estimate_value(y_pred)
            y_pred = np.squeeze(y_pred)
            print("\r  Batch({:6}/{:6})[{}]:".format(
                batch, len(valid_loader),
                ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30]
                #batch, len(train_loader),
                #('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            ), end="")
            #print(pred)
            #print(y_true.shape)
            #y_pred = y_pred.cpu().detach().numpy()
            #y = np.concatenate([y_pred, y_class])
            #print(np.concatenate([y_pred, y_class]))
            #print(y_true)
            #print(y_class)
            for i in range(len(y_pred)):
                if(y_class[i] == 0):
                    #print(0)
                    x1.append(y_pred[i])  
                    #print(x1)
                elif(y_class[i] == 1):
                    #print(1)
                    x2.append(y_pred[i])
                elif(y_class[i] == 2):
                    #print(2)
                    x3.append(y_pred[i])
                    #print(y_pred[i])
                elif(y_class[i] == 3):
                    #print(3)
                    x4.append(y_pred[i])
                #ax1.set_xlabel("Pred")
                #ax1.set_ylabel("OS")
                #ax2.set_xlabel("Pred",fontsize=18)
                #ax2.set_ylabel("Freq",fontsize=18)
                #ax2.set_ylim(0, 50)
        #ax2.hist([x1, x2, x3, x4], bins=50, color=['red', 'blue', 'green','yellow'], 
        #            label=['class 0 (0~12mo.)', 'class 1 (12~24mo.)', 'class 2 (24~36mo.)','class 3 (36~48mo.)'],histtype='bar', stacked=True)
        #x1 = torch.rand(1000)
        #x2 = torch.rand(1000)
        #x3 = torch.rand(1000)
        #x4 = torch.rand(1000)
        data_list = [x1, x2, x3, x4]


        #fig3, axes2 = plt.subplots(4,1,figsize=(4, 8))
        fig3, axes2 = plt.subplots(2,2)
        ax = axes2.ravel()  
        cm = ['red','blue','green','orange']
        #print(data_list[0])
        i = 0
        for i in range(4):
            print(i)
            #ax[i] = fig3.add_subplot(4,1,i+1)
            ax[i].hist(data_list[i],color = cm[i], bins=50,alpha = 0.5)
            
            #ax[i].set_xticks(np.arange(0,51,12))
            #ax[i].set_yticks(np.arange(0,12000,3000))
            ax[i].set_xticks(np.arange(0,51,5))
            #ax[i].set_ylim(0,100000)
            ax[i].set_title('class ' + str(i) + ' (' + str(12*i) + '~' + str(12*(i+1)) + 'months)')
            #ax[i].set_yscale('log')
            
            ax[i].set_ylabel('Frequency')
            ax[i].set_xlabel('Prediction')
        #fig3.text(0.5, 0.05, 'Prediction Value', ha='center', va='center')
        #fig3.text(0.01, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')

        """
        ax1 = fig2.add_subplot(2, 2, 1) 
        ax1.hist(x1, bins=50,  color='red', alpha = 0.5)
        ax2 = fig2.add_subplot(2, 2, 2) 
        ax2.hist(x2, bins=50,  color='blue', alpha = 0.5)
        ax3 = fig2.add_subplot(2, 2, 3) 
        ax3.hist(x3, bins=50,  color='green', alpha = 0.5)
        ax4 = fig2.add_subplot(2, 2, 4) 
        ax4.hist(x4, bins=50,  color='yellow', alpha = 0.5)
        """
        #ax2.set_xlabel("Pred",fontsize=18)
        #ax2.set_ylabel("Freq",fontsize=18)
        #ax2.set_xticks(np.arange(0,51,5))
        #ax2.set_yticks(np.arange(0,1801,200))
        #ax2.set_title('Distribution of predictions per class on validation data',fontsize=18)
        #ax2.legend(loc='upper left',fontsize=14)
        fig3.tight_layout()
        fig3.savefig(f'valid_pred_hist.png')
        #fig3.savefig(f'train_pred_hist.png')
        plt.clf()
        plt.close()
if __name__ == '__main__':
     main()
