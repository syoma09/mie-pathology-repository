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
from collections import defaultdict  # defaultdictをインポート


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
            self.__dataset += [
                (path, label, subject)  # 患者IDを追加
                for path in (root / subject).iterdir()
            ]

        random.shuffle(self.__dataset)

        self.__num_class = 4

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, label, subject = self.__dataset[item]

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

        return img, soft_labels, label, label_class, subject  # subjectを返す

def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )

    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()

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
    batch_size = 32
    num_workers = os.cpu_count() // 4

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


    net = AutoEncoder2()
    net.load_state_dict(torch.load(
        road_root / "20240612_193244" / 'state01000.pth', map_location=device)
    )
    net.dec = nn.Sequential(
        nn.Flatten(), #作っていいのかわからない
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

    optimizer = torch.optim.RAdam(net.parameters(), lr=0.001)

    LAMBDA_1 = 0.2
    LAMBDA_2 = 0.05
    START_AGE = 0
    END_AGE = 3

    criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE)
    criterion2 = nn.KLDivLoss(reduction='batchmean')

    tensorboard = SummaryWriter(log_dir='./logs', filename_suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        net.train()
        train_loss = 0.
        train_mean_loss = 0.
        train_variance_loss = 0.
        train_softmax_loss = 0.
        train_mae = 0.
        train_index = 0.
        train_loss_mae = 0.
        for batch, (x, soft_labels, y_true, y_class, subject) in enumerate(train_loader):
            optimizer.zero_grad()
            y_true, soft_labels, y_class = y_true.to(device), soft_labels.to(device), y_class.to(device)

            y_pred = net(x.to(device))

            mean_loss, variance_loss = criterion1(y_pred, y_class, device)
            softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)

            loss = mean_loss + variance_loss + softmax_loss
            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            status = np.ones(len(y_true))
            index = concordance_index(y_true.cpu().numpy(), pred, status)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            train_mean_loss += mean_loss / len(train_loader)
            train_variance_loss += variance_loss / len(train_loader)
            train_softmax_loss += softmax_loss / len(train_loader)
            train_mae += mae / len(train_loader)
            train_index += index / len(train_loader)

            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item(), softmax_loss, mean_loss, variance_loss
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

        net.eval()
        metrics = {
            'train': {
                'loss': 0.,
            }, 'valid': {
                'loss': 0.,
            }
        }

        with torch.no_grad():
            loss_mae = 0.
            mean_loss_val = 0.
            variance_loss_val = 0.
            softmax_loss_val = 0.
            valid_mae = 0.
            valid_index = 0.

            # 患者ごとの予測値を保持する辞書
            patient_predictions = defaultdict(list)
            patient_true_labels = {}

            for batch, (x, soft_labels, y_true, y_class, subjects) in enumerate(valid_loader):
                x, y_true, soft_labels, y_class = x.to(device), y_true.to(device), soft_labels.to(device), y_class.to(device)

                y_pred = net(x)
                mean_loss, variance_loss = criterion1(y_pred, y_class, device)
                softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)

                loss = mean_loss + variance_loss + softmax_loss
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                status = np.ones(len(y_true))
                index = concordance_index(y_true.cpu().numpy(), pred, status)

                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_mae += mae / len(valid_loader)
                mean_loss_val += mean_loss / len(valid_loader)
                variance_loss_val += variance_loss / len(valid_loader)
                softmax_loss_val += softmax_loss / len(valid_loader)
                valid_index += index / len(valid_loader)

                # 患者ごとの予測値を収集
                for i, subject in enumerate(subjects):
                    patient_predictions[subject].append(pred[i].item())
                    if subject not in patient_true_labels:
                        patient_true_labels[subject] = y_true[i].item()

                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(), softmax_loss, mean_loss, variance_loss
                ), end="")

            # 患者ごとの予測値を計算
            patient_avg_predictions = {subject: np.mean(preds) for subject, preds in patient_predictions.items()}

            # 結果を表示
            print("\n患者ごとの予測結果:")
            for subject in patient_avg_predictions:
                predicted = patient_avg_predictions[subject]
                actual = patient_true_labels[subject]
                print(f"患者 {subject}: 予測生存期間 = {predicted:.2f}, 実際の生存期間 = {actual:.2f}")

        print("    valid MV: {:3.3}".format(metrics['valid']['loss']))
        print('')
        print("    valid MAE: {:3.3}".format(valid_mae))
        print('')
        print("    valid INDEX: {:3.3}".format(valid_index))


        tensorboard.add_scalar('train_MV', train_loss, epoch)
        tensorboard.add_scalar('train_Mean', train_mean_loss, epoch)
        tensorboard.add_scalar('train_Variance', train_variance_loss, epoch)
        tensorboard.add_scalar('train_Softmax', train_softmax_loss, epoch)
        tensorboard.add_scalar('train_MAE', train_mae, epoch)
        tensorboard.add_scalar('train_Index', train_index, epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_Mean', mean_loss_val, epoch)
        tensorboard.add_scalar('valid_Variance', variance_loss_val, epoch)
        tensorboard.add_scalar('valid_Softmax', softmax_loss_val, epoch)
        tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        tensorboard.add_scalar('valid_Index', valid_index, epoch)


if __name__ == '__main__':
    main()
    

