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
import numpy as np
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt
from AutoEncoder import AutoEncoder, create_model
#import cv2
import numpy as np
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.stats import norm
from function import load_annotation, get_dataset_root_path
from VAE import VAE,PatchDataset


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

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
    
    # Log, epoch-model output directory
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        #"../_data/survival_time_cls/20230606_survival.csv"
        "../_data/survival_time_cls/fake_data.csv"
    ).expanduser()
    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    # Existing subjects are ignored in the function
    
    # Load annotations
    annotation = load_annotation(annotation_path)
    #print(annotation)
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    """valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )"""
    #dataset_root = Path("/mnt/cache").expanduser()/ os.environ.get('USER') / 'mie-pathology'
    z_dim = 512
    net = create_model().to(device)
    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        road_root / "20230628_153447" /'20230628_153453model00000.pth', map_location=device) #U-net
        #dataset_root / '20220720_155118model00172.pth', map_location=device)
    )
    output_and_label = []
    
    #print(net)
    with torch.no_grad():
        for batch, (x) in enumerate(train_loader):
            x  = x.to(device)
            #vector,reconstructions = net(x)
            reconstructions = net(x)
            print("\r  Batch({:6}/{:6})[{}]: ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            ), end="")
        output_and_label.append((reconstructions,x))
        org, img = output_and_label[-1]
        tmp = org[0,:,:,:]
        tmp = unnorm(tmp)
        #[横幅,縦幅,色]にする
        tmp = tmp.permute(1, 2, 0)
        #tensolからnumpyにする
        tmp = tmp.to('cpu').detach().numpy().copy()        
        #nyumpyからpilにする
        img_pil = Image.fromarray((tmp*255).astype(np.uint8))
        #画像ファイルを出力して確認します
        img_pil.save("imagefile/pred_image.png")
        tmp = img[0,:,:,:]
        tmp = unnorm(tmp)
        #[横幅,縦幅,色]にする
        tmp = tmp.permute(1, 2, 0)
        #tensolからnumpyにする
        tmp = tmp.to('cpu').detach().numpy().copy()        
        #nyumpyからpilにする
        img_pill = Image.fromarray((tmp*255).astype(np.uint8))
        #画像ファイルを出力して確認します
        img_pill.save("imagefile/true_image.png")

if __name__ == '__main__':
     main()