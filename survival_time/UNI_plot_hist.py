import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lifelines.utils import concordance_index
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import logging  # ログメッセージ用ライブラリ

from aipatho.dataset import load_annotation
from create_soft_labels import estimate_value, create_softlabel_survival_time_wise
from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    # 画像とラベルを取得して4クラスに分類
    def __getitem__(self, item):
        path, label = self.__dataset[item]

        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        # 破損した画像があるようなので一時変更
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            # エラーが発生した場合、その画像をスキップして次の画像を読み込む
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

    # 各クラスの画像枚数をカウントするメソッド
    def count_classes(self):
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
        return class_counts

#なんだこれ？
def transform_image(img):
    if isinstance(img, torch.Tensor):
        return img
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transform_image(img)

class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1024, 512)  # 1024次元（ViTの出力）から512次元に変換する線形層
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

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    patch_size = 256, 256
    stride = 256, 256

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )

    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()
    annotation = load_annotation(annotation_path)

    batch_size = 16
    num_workers = os.cpu_count() // 4

    
    """train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag=0), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )"""

    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag=1), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    # モデルの準備
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    base_model.eval()
    base_model.to(device)

    model = CustomModel(base_model)
    model.load_state_dict(torch.load("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20241008_180320/model00100.pth", map_location="cpu"), strict=True)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    logging.info("検証データの予測を開始します")
    # 検証データを使用して予測を行い、ヒストグラムをプロット
    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision用のスケーラー
    with torch.no_grad():
        x1, x2, x3, x4 = [], [], [], []
        for batch, (x, soft_labels, y_true, y_class) in enumerate(valid_loader): # valid_loaderから変更
            logging.info(f"バッチ {batch + 1} を処理中")
            x, y_true = x.to(device), y_true.to(device)
            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)

            with torch.cuda.amp.autocast():  # Mixed Precisionの自動キャスト
                outputs = model(x)
                y_pred = estimate_value(outputs)
                y_pred = np.squeeze(y_pred)

            for i in range(len(y_pred)):
                if y_true[i] < 11:
                    x1.append(y_pred[i])
                elif y_true[i] < 22:
                    x2.append(y_pred[i])
                elif y_true[i] < 33:
                    x3.append(y_pred[i])
                elif y_true[i] < 44:
                    x4.append(y_pred[i])

    logging.info("予測が完了しました。ヒストグラムをプロットします")

    data_list = [x1, x2, x3, x4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.ravel()
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        ax[i].hist(data_list[i], color=colors[i], bins=50, alpha=0.5)
        ax[i].set_xticks(np.arange(0, 51, 5))
        ax[i].set_title(f'class {i} ({12 * i}~{12 * (i + 1)} months)')
        ax[i].set_ylabel('Frequency')
        ax[i].set_xlabel('Prediction')
    fig.tight_layout()

    # ファイルパスの検証とサニタイズ
    save_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(save_dir, 'valid_pred_hist.png')
    if not os.path.commonprefix([save_dir, save_path]) == save_dir:
        raise ValueError("Invalid file path")

    fig.savefig(save_path)
    plt.clf()
    plt.close()

    # ファイルの存在を確認
    if os.path.exists(save_path):
        logging.info(f"ヒストグラムが保存されました: {save_path}")
    else:
        logging.error("ヒストグラムの保存に失敗しました")

if __name__ == '__main__':
    main()