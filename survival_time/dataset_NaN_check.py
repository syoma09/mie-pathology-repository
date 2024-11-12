import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm  # 進捗バー表示のためのライブラリ

from aipatho.dataset import load_annotation
from create_soft_labels import create_softlabel_survival_time_wise

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
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            return self.__getitem__((item + 1) % len(self.__dataset))
        
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

def check_nan_in_dataset(dataset, device):
    for idx in tqdm(range(len(dataset)), desc="Checking dataset"):
        img, soft_labels, label, label_class = dataset[idx]
        img, soft_labels, label = img.to(device), soft_labels.to(device), label.to(device)
        if torch.isnan(img).any() or torch.isnan(soft_labels).any() or torch.isnan(label).any():
            print(f"NaN found in dataset at index {idx}")
            return
    print("No NaNs found in the dataset")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path("~/data/_out/mie-pathology/").expanduser()
    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()
    annotation = load_annotation(annotation_path)

    flag = 0
    train_dataset = PatchDataset(dataset_root, annotation['train'], flag)
    valid_dataset = PatchDataset(dataset_root, annotation['valid'], flag)

    print("Checking NaNs in training dataset...")
    check_nan_in_dataset(train_dataset, device)

    print("Checking NaNs in validation dataset...")
    check_nan_in_dataset(valid_dataset, device)

if __name__ == '__main__':
    main()