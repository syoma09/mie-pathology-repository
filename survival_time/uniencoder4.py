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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

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

        return img, soft_labels, label, label_class

    def count_images_per_class(self):
        class_counts = [0] * self.__num_class
        for _, label in self.__dataset:
            if label < 11:
                class_counts[0] += 1
            elif label < 22:
                class_counts[1] += 1
            elif label < 33:
                class_counts[2] += 1
            elif label < 44:
                class_counts[3] += 1
            else:
                raise ValueError(f"Unexpected label value: {label}")
        return class_counts

def transform_image(img):
    if isinstance(img, torch.Tensor):
        return img
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transform_image(img)

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
    batch_size = 16  # バッチサイズを大きくする
    num_workers = os.cpu_count()  # num_workersを増やす
    print(f"num_workers = {num_workers}")

    flag = 0
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True  # pin_memory=Trueを追加
    )

    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True  # pin_memory=Trueを追加
    )
    
    train_dataset = PatchDataset(dataset_root, annotation['train'], flag)
    class_counts = train_dataset.count_images_per_class()
    print(f"Class counts in training dataset: {class_counts}")

    valid_dataset = PatchDataset(dataset_root, annotation['valid'], flag)
    class_counts = valid_dataset.count_images_per_class()
    print(f"Class counts in validation dataset: {class_counts}")

    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    login(token)

    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class CustomModel(nn.Module):
        def __init__(self, base_model):
            super(CustomModel, self).__init__()
            self.base_model = base_model
            self.fc = nn.Linear(1024, 512)
            self.additional_layers = nn.Sequential(
                nn.Flatten(),
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

        def forward(self, x):
            features = self.base_model(x)
            features = self.fc(features)
            output = self.additional_layers(features)
            return output

    model = CustomModel(model)
    model = model.to(device)
    
    dummy_input = torch.randn(32, 3, 256, 256).to(device)
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

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

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        model.train()
        train_loss = 0.
        train_mean_loss = 0.
        train_variance_loss = 0.
        train_softmax_loss = 0.
        train_mae = 0.
        train_index = 0.
        train_loss_mae = 0.
        for batch, (x, soft_labels, y_true, y_class) in enumerate(train_loader):
            optimizer.zero_grad()
            y_true, soft_labels, y_class = y_true.to(device), soft_labels.to(device), y_class.to(device)

            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(x)

                y_pred = outputs
                if y_pred.dim() == 1:
                    y_pred = y_pred.unsqueeze(0)
                p = torch.softmax(y_pred, dim=-1)

                mean_loss, variance_loss = criterion1(p, y_class, device)
                softmax_loss = criterion2(torch.log_softmax(y_pred, dim=-1), soft_labels)

                loss = mean_loss + variance_loss + softmax_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            status = np.ones(len(y_true))
            index = concordance_index(y_true.cpu().numpy(), pred, status)

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
        torch.save(model.state_dict(), log_root / f"model{epoch:05}.pth")

        model.eval()
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
            for batch, (x, soft_labels, y_true, y_class) in enumerate(valid_loader):
                x, y_true, soft_labels, y_class = x.to(device), y_true.to(device), soft_labels.to(device), y_class.to(device)

                x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
                x = torch.stack([transform_image(img) for img in x]).to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(x)

                    y_pred = outputs
                    if y_pred.dim() == 1:
                        y_pred = y_pred.unsqueeze(0)
                    p = torch.softmax(y_pred, dim=-1)

                    mean_loss, variance_loss = criterion1(p, y_class, device)
                    softmax_loss = criterion2(torch.log_softmax(y_pred, dim=-1), soft_labels)

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

                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(), softmax_loss, mean_loss, variance_loss
                ), end="")

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
