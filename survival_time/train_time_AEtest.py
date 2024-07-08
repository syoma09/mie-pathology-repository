#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
import numpy as np
from PIL import Image

#from AutoEncoder import create_model
from survival import  create_model
#from lifelines.utils import concordance_index
from aipatho.dataset import load_annotation
from aipatho.metrics import MeanVarianceLoss
from create_soft_labels import estimate_value, create_softlabel_tight, create_softlabel_survival_time_wise

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import create_dataset
# from aipatho.metrics.label import GaussianSoft


#榊原
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(
            #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #)
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.__dataset = []

        #CNN
        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
            ]


        # print((self.__dataset))
        # self.__dataset = list(itertools.chain.from_iterable(self.__dataset))
        random.shuffle(self.__dataset)
        # print((self.__dataset))
        # print(len(self.__dataset))
        print("a")
  
        self.__num_class = 4

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
        #self.l = self.__dataset[item]
        #print(self.l)
        #path, label = self.l[item]

        # CNN
        path, label = self.__dataset[item]
        if os.path.isdir(path):
            # ディレクトリの場合はエラーメッセージを出してスキップ
            return self.__getitem__((item + 1) % len(self.__dataset))
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        # label /= 90.
        
        if label < 11:
            label_class = 0
        elif label < 22:
            label_class = 1
        elif label < 33:
            label_class = 2
        elif label < 44:
            label_class = 3

        # Tensor
        label = torch.tensor(label, dtype=torch.float)
        
        # Soft label
        num_classes = 4
        # soft_labels = GaussianSoft(num_classes)(label_class)    # basic
        # soft_labels = create_softlabel_tight(label,num_classes)#tight
        soft_labels = create_softlabel_survival_time_wise(label, num_classes)#survivaltime_wise
        
        return img, soft_labels, label, label_class     # CNN
        #return img_group, soft_labels, label, label_class #Transformer

    def pull_group(self, item: int) -> ([str], [float]):
        return (
           [path for path, _  in self.__dataset[item]],
           [label for _, label  in self.__dataset[item]] 
        )

    def pathtoimg(self,path_group):
        return [Image.open(path).convert('RGB') for path in path_group] 


def main():
    patch_size = 256,256
    stride = 256,256
    index = 2

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )
    dataset_root_not = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.SEVERE
    )
    
    # Log, epoch-model output directory
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        #"../_data/survival_time_cls/20220726_cls/cv0.csv"
        #"../_data/survival_time_cls/20220726_cls/cv1.csv"
        #"../_data/survival_time_cls/20220726_cls/cv2.csv"
        #"../_data/survival_time_cls/20220726_cls.csv"
        "_data/survival_time_cls/20220726_cls.csv"
        #"../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        #"../_data/survival_time_cls/fake_data.csv"
    ).expanduser()

    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None,
        target=TumorMasking.FULL
    )
    # create_dataset(
    #     src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
    #     dst=dataset_root_not,
    #     annotation=annotation_path,
    #     size=patch_size, stride=stride,
    #     index=index, region=None,
    #     target = TumorMasking.SEVERE
    # )

    # Load annotations
    annotation = load_annotation(annotation_path)

    epochs = 1000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 4   # For SMT
    # Load train/valid yaml
    # with open(src / "survival_time.yml", "r") as f:
    #     yml = yaml.safe_load(f)

    # データ読み込み
    flag = 0
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag), batch_size=batch_size,shuffle = True,
        num_workers=num_workers,drop_last = True
    )
    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=batch_size,
        num_workers=num_workers,drop_last = True
    )


    # AE
    net = create_model() 
    
    net.load_state_dict(torch.load(
        road_root / "20240612_193244" /'state01000.pth', map_location=device) #AE                                 
    )

    net.dec = nn.Sequential( #AE
        nn.Flatten(), #入力サイズ合わせるために勝手に層作っちゃっていいのか？
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
    
    net = net.to(device)
    

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    # optimizer_ext = torch.optim.RAdam(ext.parameters(), lr=0.0001)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.001)

    LAMBDA_1 = 0.2
    #LAMBDA_1 = 1.
    LAMBDA_2 = 0.05
    #LAMBDA_2 = 1.
    START_AGE = 0
    END_AGE = 3
    VALIDATION_RATE= 0.1
    
    criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE)
    #criterion2 = nn.CrossEntropyLoss() #hard label
    criterion2 = nn.KLDivLoss(reduction='batchmean') # soft label

    tensorboard = SummaryWriter(log_dir='./logs',filename_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    print(net)

    seq_len = 8
    num_classes = 4

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        train_mean_loss = 0.
        train_variance_loss = 0.
        train_softmax_loss = 0.
        train_mae = 0.
        train_index=0.
        train_loss_mae = 0.
        for batch, (x, soft_labels, y_true, y_class) in enumerate(train_loader):
            optimizer.zero_grad()
            y_true, soft_labels, y_class = y_true.to(device), soft_labels.to(device), y_class.to(device)
            
            y_pred = net(x.to(device))

            mean_loss, variance_loss = criterion1(y_pred, y_class, device)
            
            # Soft label
            # print(soft_labels)
            softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)
            
            #hard label
            #softmax_loss = criterion2(y_pred, y_class)
            #print(softmax_loss)
            
            #loss = softmax_loss
            loss = mean_loss + variance_loss +  softmax_loss
            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            # print('')
            # print(mae)
            # print('')
            status = np.ones(len(y_true))
            
            index = concordance_index(y_true.cpu().numpy(), pred, status)  
            
            #print(loss)
            #loss = criterion(y_pred,y_true)
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            # Logging
            train_loss += loss.item() / len(train_loader)
            train_mean_loss += mean_loss / len(train_loader)
            train_variance_loss += variance_loss / len(train_loader)
            train_softmax_loss += softmax_loss / len(train_loader)
            train_mae += mae / len(train_loader)
            
            train_index += index / len(train_loader)
            
            #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}" .format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                #loss.item()
                loss.item(),softmax_loss,mean_loss,variance_loss
            ), end="")
            print(f" {mae:.3}", end="")

        print("    train MV: {:3.3}".format(train_loss))
        print('')
        print("    train MAE: {:3.3}".format(train_mae))
        print('')
        print("    train INDEX: {:3.3}".format(train_index))
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), log_root / f"model{epoch:05}.pth")
        # Switch to evaluation mode
        net.eval()
        # On training data
        # Initialize validation metric values
        metrics = {
            'train': {
                'loss': 0.,
            }, 'valid': {
                'loss': 0.,
            }
        }
        # Calculate validation metrics
        with torch.no_grad():
            loss_mae = 0.
            mean_loss_val = 0.
            variance_loss_val = 0.
            softmax_loss_val = 0.
            valid_mae = 0.
            valid_index = 0.
            for batch, (x, soft_labels, y_true,y_class) in enumerate(valid_loader):
                x, y_true, soft_labels, y_class = x.to(device), y_true.to(device) ,soft_labels.to(device), y_class.to(device)

                y_pred = net(x)  # Prediction
                #yt_one = torch.from_numpy(OnehotEncording(y_class)).to(device)
                mean_loss, variance_loss = criterion1(y_pred, y_class,device)
                
                #soft label
                softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)

                #hard label
                #softmax_loss = criterion2(y_pred, y_class)
                
                loss = mean_loss + variance_loss +  softmax_loss
                #loss = softmax_loss
                
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                status = np.ones(len(y_true))
                index = concordance_index(y_true.cpu().numpy(), pred, status)
                #loss = criterion(y_pred,y_true)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_mae += mae  / len(valid_loader)
                mean_loss_val += mean_loss / len(valid_loader)
                variance_loss_val += variance_loss / len(valid_loader)
                softmax_loss_val += softmax_loss / len(valid_loader)
                valid_index += index / len(valid_loader)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(),softmax_loss,mean_loss,variance_loss
                    #loss.item()
                ), end="")
        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        print("    valid MV: {:3.3}".format(metrics['valid']['loss']))
        print('')
        print("    valid MAE: {:3.3}".format(valid_mae))
        print('')
        print("    valid INDEX: {:3.3}".format(valid_index))
        
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard'''
        #tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('train_MV', train_loss, epoch)
        tensorboard.add_scalar('train_Mean', train_mean_loss, epoch)
        tensorboard.add_scalar('train_Variance', train_variance_loss, epoch)
        tensorboard.add_scalar('train_Softmax', train_softmax_loss, epoch)
        tensorboard.add_scalar('train_MAE', train_mae, epoch)
        tensorboard.add_scalar('train_Index', train_index, epoch)
        #tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_Mean', mean_loss_val, epoch)
        tensorboard.add_scalar('valid_Variance', variance_loss_val, epoch)
        tensorboard.add_scalar('valid_Softmax', softmax_loss_val, epoch)
        tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        tensorboard.add_scalar('valid_Index', valid_index, epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)


if __name__ == '__main__':
    main()
