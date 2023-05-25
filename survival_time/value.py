import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import yaml
import math
import random
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import AutoEncoder
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from function import load_annotation, get_dataset_root_path
from train_time import PatchDataset,estimate_value
from Unet import Generator
from cnn.metrics import ConfusionMatrix

device = 'cuda:1'

def main():
    patch_size = 256,256
    stride = 256,256
    index = 2
    # patch_size = 256, 256
    
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )
    
    
    # Log, epoch-model output directory
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        "../_data/survival_time_cls/20220726_cls.csv"
        #"../_data/survival_time_cls/fake_data.csv"
    ).expanduser()
    # Create dataset if not exists
    """if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)"""
    # Existing subjects are ignored in the function
    """create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )"""
    # Load annotations
    annotation = load_annotation(annotation_path)
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    epochs = 10000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    # Load train/valid yaml
    '''with open(src / "survival_time.yml", "r") as f:
        yml = yaml.safe_load(f)'''

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    z_dim = 512
    #net = UNet_2D().to(device)
    net = Generator().to(device)
    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        road_root / "20230516_162742" /'model00149.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #dataset_root / '20220720_155118model00172.pth', map_location=device)
        
    )
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU() 
    net = nn.Sequential(
        net,
        nn.Sequential(
        nn.BatchNorm1d(512),
        #nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        #nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        #nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )
        #nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        #road_root / "20230523_144059" /'20230523_144107model00005.pth', map_location=device) #U-net 
        road_root / "20230524_152733" /'model000490.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #dataset_root / '20220720_155118model00172.pth', map_location=device)
        
    )
    net = net.to(device)
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
    output_and_label = []
    y_true_plot = []
    valid_pred_plot = []
    train_pred_plot = []
    with torch.no_grad():
            for batch, (x, y_true, y_class) in enumerate(train_loader):
                x, y_true, y_class = x.to(device), y_true.to(device) ,y_class.to(device)
                y_pred = net(x)  # Prediction
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                for k  in range(len(y_pred)):
                    y_true_plot.append(y_true[k].cpu().numpy().tolist())
                    train_pred_plot.append(pred[k].tolist())
                    output_and_label.append((train_pred_plot, y_true_plot))
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
            fig, ax = plt.subplots()
            plt.xticks(range(0, 50, 5))
            # y軸を0〜8まで3刻みで表示
            plt.yticks(range(0, 50, 5))
            for i in range(len(train_pred_plot)):
                print(f'i : {i}')
                #ax.plot(output_and_label[0],output_and_label[1],'.')
                ax.plot(train_pred_plot[i],y_true_plot[i],'.')
                fig.savefig(f'train_valueplot.png')
                ax.set_xlabel("Pred")
                ax.set_ylabel("True")
            y_true_plot = []
            for batch, (x, y_true, y_class) in enumerate(valid_loader):
                x, y_true, y_class = x.to(device), y_true.to(device) ,y_class.to(device)
                y_pred = net(x)  # Prediction
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                for k  in range(len(y_pred)):
                    y_true_plot.append(y_true[k].cpu().numpy().tolist())
                    valid_pred_plot.append(pred[k].tolist())
                    output_and_label.append((valid_pred_plot, y_true_plot))
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
            fig, ax = plt.subplots()
            plt.xticks(range(0, 50, 5))
            # y軸を0〜8まで3刻みで表示
            plt.yticks(range(0, 50, 5))
            for i in range(len(valid_pred_plot)):
                print(f'i : {i}')
                #ax.plot(output_and_label[0],output_and_label[1],'.')
                ax.plot(valid_pred_plot[i],y_true_plot[i],'.')
                fig.savefig(f'valid_valueplot.png')
                ax.set_xlabel("Pred")
                ax.set_ylabel("True")
if __name__ == '__main__':
    main()