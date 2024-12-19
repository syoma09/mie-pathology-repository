#TCGA_train_valid_44.csvで学習したuniモデルを使用して、TCGAのデータセットを予測する
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import logging  # ログメッセージ用ライブラリ
from PIL import Image
import timm

from aipatho.dataset import load_annotation
from create_soft_labels import create_softlabel_survival_time_wise
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

# カスタムモデルの定義
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1024, 512)  # 1024次元（ViTの出力）から512次元に変換する線形層
        self.additional_layers = nn.Sequential(
            nn.Flatten(),  # 必要に応じて使用
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
        # base_model(UNI)のパラメータを固定
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        features = self.fc(features)
        output = self.additional_layers(features)
        return output

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    patch_size = 256, 256
    stride = 256, 256

    dataset_root = Path(
        "/net/nfs3/export/home/sakakibara/data/TCGA_patch_image/" # TCGAのデータセットのパス
    )

    annotation_path = Path("_data/survival_time_cls/TCGA_train_valid_44.csv").expanduser()
    annotation = load_annotation(annotation_path)

    batch_size = 64
    num_workers = os.cpu_count() // 4

    # 検証データセットのロード
    valid_dataset = PatchDataset(dataset_root, annotation['valid'], flag=1)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    # モデルの定義とロード
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model = CustomModel(base_model)
    model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20241114_145205uniencoder3/model00046.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    logging.info("検証データの予測を開始します")
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for batch, (x, soft_labels, y_true, y_class) in enumerate(valid_loader):
            if batch >= 5:  # 最初の5バッチだけを処理　この二行は必要でなければコメント
                break
            logging.info(f"バッチ {batch + 1}/{len(valid_loader)} を処理中")
            x, y_true = x.to(device), y_true.to(device)
            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)

            # モデルによる予測
            outputs = model(x)
            predictions = outputs.cpu().numpy()

            # 予測結果と正解ラベルを収集
            all_predictions.extend(predictions)
            all_labels.extend(y_true.cpu().numpy())

    logging.info("予測が完了しました。結果を表示します")

    # 予測結果の一部を表示
    for i in range(5):
        print(f"予測: {all_predictions[i]}, 実際のラベル: {all_labels[i]}")

if __name__ == '__main__':
    main()