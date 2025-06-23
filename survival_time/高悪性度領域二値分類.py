#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#uniモデルでメインの学習コード
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
from torchvision import transforms
import numpy as np
from PIL import Image
from lifelines.utils import concordance_index

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

from survival import create_model
from aipatho.dataset import load_annotation
from aipatho.metrics import MeanVarianceLoss
from create_soft_labels import estimate_value, create_softlabel_tight, create_softlabel_survival_time_wise

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import create_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []

        for subject, label in annotations:
            patchlist_path = root / subject / 'patchlist' / 'patchlist_severe.csv'
            if patchlist_path.exists():
                with open(patchlist_path, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        _, _, _, _, path, severe = line.strip().split(',')
                        self.__dataset.append((path, int(severe)))

        random.shuffle(self.__dataset)

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, severe = self.__dataset[item]

        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            return self.__getitem__((item + 1) % len(self.__dataset))
        
        img = self.transform(img)
        severe = torch.tensor(severe, dtype=torch.float)
        return img, severe

def print_gpu_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2
    
    # 三重大学のデータセットのパス
    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )
    
    """
    dataset_root = Path(
        "/net/nfs3/export/home/sakakibara/data/TCGA_patch_image/" #TCGAのデータセットのパスはこっち
    )
    """

    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "uniencoder3")
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser() #三重データの場合
    #annotation_path = Path("_data/survival_time_cls/TCGA_train_valid_44.csv").expanduser() #TCGAデータの場合

    create_dataset(
        src=Path("/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None,
        target=TumorMasking.FULL
    )
    
    annotation = load_annotation(annotation_path)

    epochs = 1000
    batch_size = 16
    num_workers = os.cpu_count() // 4
    print(f"num_workers = {num_workers}")

    flag = 0
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    # 環境変数からトークンを取得
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    # Hugging Faceにログイン
    login(token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # モデルのダウンロードと準備
    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    
    # ダウンロードしたファイルの存在確認
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # ViT-16/Lモデルの作成、num_classes=0は特徴抽出モード
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    base_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    base_model.eval()
    base_model.to(device)

    # 画像の前処理 入力に合うように
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # カスタムモデルの定義
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1024, 512)  # 1024次元（ViTの出力）から512次元に変換する線形層
        # 2値分類のための追加の線形層
        self.additional_layers = nn.Sequential(
            nn.Flatten(),  # 必要に応じて使用
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1, bias=True)  # 2値分類のための出力層
        )
        # base_model(UNI)のパラメータを固定
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        features = self.fc(features)
        output = self.additional_layers(features)
        return output

def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2
    
    # 三重大学のデータセットのパス
    dataset_root = Path(
        "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/"
    )
    
    # TCGAのデータセットのパスはコメントアウト
    """
    dataset_root = Path(
        "/net/nfs3/export/home/sakakibara/data/TCGA_patch_image/"
    )
    """

    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "uniencoder3")
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()  # 三重データの場合

    # スルーされるはず　これ確認
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None,
        target=TumorMasking.FULL
    )
    
    annotation = load_annotation(annotation_path)

    epochs = 1000
    batch_size = 16  # 適切な値に変更する
    num_workers = os.cpu_count() // 4
    print(f"num_workers = {num_workers}")

    flag = 0
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )


    # 環境変数からトークンを取得
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    # Hugging Faceにログイン
    login(token)  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # モデルのダウンロードと準備
    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    
    # ダウンロードしたファイルの存在確認
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # ViT-16/Lモデルの作成、num_classes=0は特徴抽出モード
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)  # map_location="cpu"から変更
    model.eval()
    model.to(device)

    # 画像の前処理 入力に合うように
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # カスタムモデルを準備
    model = CustomModel(model)
    model = model.to(device)
    

    print("モデルロード後:")
    print_gpu_memory_usage()
    # モデルの出力次元を確認するためのコード
    dummy_input = torch.randn(32, 3, 256, 256).to(device)  # ダミー入力
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001)  # 学習率を0.001から変更

    criterion = nn.BCEWithLogitsLoss()  # 2値分類のための損失関数

    tensorboard = SummaryWriter(log_dir='./logs', filename_suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))  # tensorboardのログを保存するディレクトリを指定
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    # Mixed precision training の準備
    scaler = torch.cuda.amp.GradScaler()

    # トレーニングループ
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        model.train()
        train_loss = 0.
        for batch, (x, y_true) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y_true = y_true.to(device).float()

            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs.squeeze(), y_true)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() / len(train_loader)

            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")

        print("    train loss: {:3.3}".format(train_loss))
        print('')
        print('    Saving model...')
        torch.save(model.state_dict(), log_root / f"model{epoch:05}.pth")

        # 検証ループ
        model.eval()
        valid_loss = 0.
        with torch.no_grad():
            for batch, (x, y_true) in enumerate(valid_loader):
                x = x.to(device)
                y_true = y_true.to(device).float()

                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    loss = criterion(outputs.squeeze(), y_true)

                valid_loss += loss.item() / len(valid_loader)

                print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item()
                ), end="")

        print("    valid loss: {:3.3}".format(valid_loss))

        # tensorboardに書き込み
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', valid_loss, epoch)

if __name__ == '__main__':
    main()