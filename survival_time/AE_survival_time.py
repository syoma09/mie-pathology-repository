#患者ごとに、AEを用いた生存期間予測の結果を算出するコード。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path
import random
import collections
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
import numpy as np
from PIL import Image
from lifelines.utils import concordance_index

#from AutoEncoder import create_model
from aipatho.model.autoencoder2  import AutoEncoder2
from survival import  create_model
#from lifelines.utils import concordance_index
from aipatho.dataset import load_annotation
from aipatho.metrics import MeanVarianceLoss
from create_soft_labels import estimate_value, create_softlabel_tight, create_softlabel_survival_time_wise

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import create_dataset
# from aipatho.metrics.label import GaussianSoft

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []

        for subject, label in annotations:
            self.__dataset += [
                (path, label)
                for path in (root / subject).iterdir()
            ]

        random.shuffle(self.__dataset)

        self.__num_class = 4

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, label = self.__dataset[item]

        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        if label < 11:
            label_class = 0
        elif label < 22:
            label_class = 1
        elif label < 33:
            label_class = 2
        elif label < 44:
            label_class = 3

        label = torch.tensor(label, dtype=torch.float)
        num_classes = 4
        soft_labels = create_softlabel_survival_time_wise(label, num_classes)

        # 患者IDを取得
        patient_id = path.parts[-2]  # ディレクトリ名を患者IDとして使用

        return img, soft_labels, label, label_class, patient_id



# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# モデルのパス
model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20240708_151949/model00806.pth" #パッチごとならMAE7.361のエポック

# データセットの準備
patch_size = 256, 256
stride = 256, 256
dataset_root = get_cache_dir(
    patch=patch_size,
    stride=stride,
    target=TumorMasking.FULL
)

annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()
annotation = load_annotation(annotation_path)

valid_loader = torch.utils.data.DataLoader(
    PatchDataset(dataset_root, annotation['valid'], flag=1),
    batch_size=32,
    num_workers=4,
    drop_last=False
)

# モデルの構築
net = AutoEncoder2()
net.dec = torch.nn.Sequential(  # デコーダ部分を定義
    torch.nn.Flatten(),
    torch.nn.Linear(512, 512, bias=True),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 512, bias=True),
    torch.nn.BatchNorm1d(512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 4, bias=True),
)

# 学習済みモデルのロード（デコーダ部分を変えてから。）
net.load_state_dict(torch.load(model_path, map_location=device))
net = net.to(device)
net.eval()

"""# 検証の実行
with torch.no_grad():
    total_mae = 0.0
    total_index = 0.0
    for batch, (x, soft_labels, y_true, y_class) in enumerate(valid_loader):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = net(x)

        # 推定値を計算
        pred = estimate_value(y_pred)  # ここで NumPy 配列が返されると仮定
        pred = np.squeeze(pred)  # NumPy 配列に対して np.squeeze を適用
        y_true_np = y_true.cpu().numpy()  # y_true を NumPy 配列に変換

        # MAE計算
        mae = np.absolute(pred - y_true_np).mean()
        total_mae += mae

        # C-index計算
        status = np.ones(len(y_true_np))  # 全てのデータが観測されていると仮定
        index = concordance_index(y_true_np, pred, status)
        total_index += index

    # 平均MAEとC-indexを出力
    print(f"Validation MAE: {total_mae / len(valid_loader):.3f}")
    print(f"Validation C-index: {total_index / len(valid_loader):.3f}")"""

# 検証の実行（患者ごとに予測）
with torch.no_grad():
    total_mae = 0.0
    total_index = 0.0
    patient_results = collections.defaultdict(list)  # 患者ごとの結果を保存する辞書

    for batch, (x, soft_labels, y_true, y_class, patient_ids) in enumerate(valid_loader):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = net(x)

        # 推定値を計算
        pred = estimate_value(y_pred)  # 推定値を計算
        pred = np.squeeze(pred)  # NumPy 配列に変換
        y_true_np = y_true.cpu().numpy()  # y_true を NumPy 配列に変換

        # 患者ごとの結果を保存
        for patient_id, true_label, predicted_label in zip(patient_ids, y_true_np, pred):
            patient_results[patient_id].append({
                "true_label": true_label,
                "predicted_label": predicted_label
            })

        # MAE計算
        mae = np.absolute(pred - y_true_np).mean()
        total_mae += mae

        # C-index計算
        status = np.ones(len(y_true_np))  # 全てのデータが観測されていると仮定
        index = concordance_index(y_true_np, pred, status)
        total_index += index

    # 平均MAEとC-indexを出力
    print(f"Validation MAE: {total_mae / len(valid_loader):.3f}")
    print(f"Validation C-index: {total_index / len(valid_loader):.3f}")

    # 患者ごとの予測結果を出力
    print("\nPatient-wise Results:")
    for patient_id, results in patient_results.items():
        true_labels = [r["true_label"] for r in results]
        predicted_labels = [r["predicted_label"] for r in results]
        avg_true_label = np.mean(true_labels)  # 真のラベルの平均
        avg_predicted_label = np.mean(predicted_labels)  # 予測ラベルの平均
        print(f"Patient {patient_id}: True Label = {avg_true_label:.3f}, Predicted Label = {avg_predicted_label:.3f}")