import torch
import numpy as np
from collections import defaultdict
from lifelines.utils import concordance_index
from pathlib import Path
from aipatho.model.autoencoder2 import AutoEncoder2
from create_soft_labels import estimate_value
from aipatho.dataset import load_annotation
from aipatho.utils.directory import get_cache_dir
from aipatho.svs import TumorMasking
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import random
import os

# データセットクラス
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []

        for subject, label in annotations:
            self.__dataset += [
                (path, label, subject)  # 患者IDを追加
                for path in (root / subject).iterdir()
            ]

        random.shuffle(self.__dataset)

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, label, subject = self.__dataset[item]

        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        img = Image.open(path).convert('RGB')
        img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float)
        return img, label, subject

# テスト用関数
def test():
    # デバイスの設定
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # モデルの準備
    net = AutoEncoder2()
    net.dec = torch.nn.Sequential(
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
    #road_root = Path("~/data/_out/mie-pathology/").expanduser()
    #net.load_state_dict(torch.load(
    #    road_root / "20240612_193244" / 'state01000.pth', map_location=device))  # 保存済みモデルをロード
    net.load_state_dict(torch.load(
        "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20240925_045157/model00425.pth", map_location=device))  # 保存済みモデルをロード
    net = net.to(device)
    net.eval()

    # データセットとデータローダーの準備
    patch_size = 256, 256
    stride = 256, 256
    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )
    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()
    annotation = load_annotation(annotation_path)

    flag = 1
    valid_loader = DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=32,
        num_workers=4, drop_last=True
    )

    # 患者ごとの予測値を保持する辞書
    patient_predictions = defaultdict(list)
    patient_true_labels = {}

    with torch.no_grad():
        for batch, (x, y_true, subjects) in enumerate(valid_loader):
            x, y_true = x.to(device), y_true.to(device)

            # モデルの予測
            y_pred = net(x)
            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)

            # 患者ごとの予測値を収集
            for i, subject in enumerate(subjects):
                patient_predictions[subject].append(pred[i].item())
                if subject not in patient_true_labels:
                    patient_true_labels[subject] = y_true[i].item()

    # 患者ごとの予測値を計算
    patient_avg_predictions = {subject: np.mean(preds) for subject, preds in patient_predictions.items()}

    # 結果を表示
    print("\n患者ごとの予測結果:")
    for subject in patient_avg_predictions:
        predicted = patient_avg_predictions[subject]
        actual = patient_true_labels[subject]
        print(f"患者 {subject}: 予測生存期間 = {predicted:.2f}, 実際の生存期間 = {actual:.2f}")

if __name__ == '__main__':
    test()