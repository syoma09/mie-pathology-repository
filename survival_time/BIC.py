#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import torch
import torch.utils.data

from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from VAE import VAE, PatchDataset

device = 'cuda:0'

def classtaling(X):
        # クラスタ数 1～20をフィティングして試す
        rangeary = np.arange(100, 101)
        gms_per_k = []
        for k in rangeary:
            gm = GaussianMixture(n_components=k, n_init=10, random_state=42)
            gm.fit(X)
            gms_per_k.append(gm)
            print(k)
            #print(gm.fit(X).bic(X))
        bics = [model.bic(X) for model in gms_per_k]	# 入力Xの現在モデルのベイズ情報量基準      

        # コンソール表示
        for k, b in zip(rangeary, bics):
            print('{:2d}: {:.f} '.format(k, b))
        
        # シルエットスコアグラフ
        #plt.title('')
        plt.plot(rangeary, bics, "bo-", color="blue", label="BIC")
        plt.xlabel("k", fontsize=14)
        plt.ylabel("BIC", fontsize=14)
        """plt.annotate('Minimum',            # 
                    xy=(3, bics[2]),
                    xytext=(0.35, 0.6),
                    textcoords='figure fraction',
                    fontsize=14,
                    arrowprops=dict(facecolor='black', shrink=0.1))"""
        plt.legend()
        plt.savefig(f"./BIC_plot.png")


def main():
    #patch_size = 512,512
    patch_size = 256, 256
    stride = 256, 256
    #stride = 128,128
    index = 2
    # patch_size = 256, 256
    
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index=index
    )

    dataset_root_not = get_dataset_root_not_path(
        patch_size=patch_size,
        stride=stride,
        index=index
    )

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not log_root.exists():
        log_root.mkdir(parents=True, exist_ok=True)
    
    annotation_path = Path(
        #"../_data/survival_time_cls/20220725_aut1.csv"
        "../_data/survival_time_cls/20230606_survival.csv"
    ).expanduser()
    
    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    annotation = load_annotation(annotation_path)
    
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    '''
    モデルの構築
    '''
    z_dim = 512
    net = VAE(z_dim).to(device)
    torch.autograd.set_detect_anomaly(True) 
    #optimizer = torch.optim.RAdam(list(net.parameters())+list(G_estimate.parameters()), lr=0.001, weight_decay=0.0001)
    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        road_root / "20230614_151005" /'model00000.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #dataset_root / '20220720_155118model00172.pth', map_location=device)
        
    )
    with torch.no_grad():
        list_vector = []

        for batch, (x) in enumerate(train_loader):
            input = x.to(device)
            vector, _ = net(input)
            list_vector.append(vector.cpu().numpy())
            print("\r  Batch({:6}/{:6})[{}]:".format(
                  batch, len(train_loader),
                  ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30]
            ), end="")
        np_vector = list_vector[0]
        for i in range(1,len(list_vector)):
            np_vector = np.vstack((np_vector,list_vector[i]))
        classtaling(np.array(np_vector))

if __name__ == "__main__":
    main()