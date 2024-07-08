#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
#import sys

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms

from aipatho.svs import TumorMasking
from aipatho.model import AutoEncoder2
from aipatho.utils.directory import get_logdir, get_cache_dir
from aipatho.dataset import PatchDataset, load_annotation, create_dataset
from aipatho.metrics.label import TimeToTime
from aipatho.utils.directory import get_cache_dir

# デバイスの選択
device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_images(input_images, reconstructed_images, save_dir, prefix='img'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, (input_img, recon_img) in enumerate(zip(input_images, reconstructed_images)):
        input_img = transforms.ToPILImage()(input_img.cpu())
        recon_img = transforms.ToPILImage()(recon_img.cpu())

        input_img.save(os.path.join(save_dir, f"{prefix}_input_{i}.png"))
        recon_img.save(os.path.join(save_dir, f"{prefix}_recon_{i}.png"))

def main():
    # モデルのパラメータを読み込む
    model_path = "/net/nfs2/export/home/sakakibara/data/_out/mie-pathology/20240612_193244/state01288.pth"
    net = AutoEncoder2().to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # データセットの準備
    patch_size = 512, 512
    stride = 512, 512
    target = TumorMasking.FULL

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=target
    )

    annotation_path = Path("_data/survival_time_cls/20220413_aut2.csv")
    annotation = load_annotation(annotation_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_dataset = PatchDataset(dataset_root, annotation['valid'], transform=transform, labeler=TimeToTime())
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=5,  # 5枚を取得
        shuffle=True
    )

    # 入力画像と再構成画像の保存
    save_dir = "/net/nfs2/export/home/sakakibara/root/workspace/mie-pathology-repository/saikousei"
    with torch.no_grad():
        for batch, (x, _) in enumerate(valid_loader):
            x = x.to(device)
            y_pred = net(x)
            save_images(x, y_pred, save_dir, prefix=f"batch{batch}")
            break  # 1バッチのみ処理

if __name__ == '__main__':
    main()
