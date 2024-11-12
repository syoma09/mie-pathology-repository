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
import logging  # ログメッセージ用ライブラリ

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

    dataset = PatchDataset(dataset_root, annotation['train'], flag=0)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    logging.info("正解データのヒストグラムを作成します")
    with torch.no_grad():
        all_labels = []
        for batch, (x, soft_labels, y_true, y_class) in enumerate(train_loader):
            logging.info(f"バッチ {batch + 1} を処理中")
            x, y_true = x.to(device), y_true.to(device)
            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)

            # 正解ラベルを収集
            y_true = y_true.cpu().numpy()

            # y_trueの一部をターミナルに表示
            print(f"バッチ {batch + 1} の正解ラベル: {y_true[:5]}")  # 最初の5つのラベルを表示

            all_labels.extend(y_true)

    logging.info("ヒストグラムをプロットします")

    plt.figure(figsize=(10, 6))
    plt.hist(all_labels, bins=50, color='blue', alpha=0.7)
    plt.xticks(np.arange(0, 51, 5))
    plt.title('True Labels Histogram')
    plt.xlabel('True Label')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # ファイルパスの検証とサニタイズ
    save_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(save_dir, 'true_label_hist.png')
    if not os.path.commonprefix([save_dir, save_path]) == save_dir:
        raise ValueError("Invalid file path")

    plt.savefig(save_path)
    plt.clf()
    plt.close()

    # ファイルの存在を確認
    if os.path.exists(save_path):
        logging.info(f"ヒストグラムが保存されました: {save_path}")
    else:
        logging.error("ヒストグラムの保存に失敗しました")

    # 各クラスの画像枚数を計算して表示
    class_counts = dataset.count_classes()
    print(f"クラス0の画像枚数: {class_counts[0]}")
    print(f"クラス1の画像枚数: {class_counts[1]}")
    print(f"クラス2の画像枚数: {class_counts[2]}")
    print(f"クラス3の画像枚数: {class_counts[3]}")

if __name__ == '__main__':
    main()