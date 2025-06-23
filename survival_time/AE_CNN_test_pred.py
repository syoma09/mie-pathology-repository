# 20250617 AE_CNN_3bunkatuで学習したモデルを用いて、テストデータセットの生存期間予測を行うコード。スライド単位の値を出力(mie_pathology全データ入ってなかったから、全データ学習しきれてない)
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from lifelines.utils import concordance_index
from aipatho.model.autoencoder2 import AutoEncoder2

import os
import random
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import transforms
from PIL import Image

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.metrics import MeanVarianceLoss

# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

# 12ヶ月ごと0-48ヶ月で4クラス
def create_softlabel_survival_time_wise(label, num_classes=4):
    """
    0-12, 12-24, 24-36, 36-48 の4クラスで線形補間softlabelを作成
    """
    soft_labels = torch.zeros(num_classes)
    for j in range(num_classes-1):
        k = j+1
        d_label_l = 12.0 * j + 6.0
        d_label_r = 12.0 * k + 6.0
        if d_label_l < label <= d_label_r:
            left = label - d_label_l
            right = d_label_r - label
            left_ratio = right / (left + right)
            right_ratio = left / (left + right)
            soft_labels[j] = left_ratio
            soft_labels[k] = right_ratio
            break
    if label <= 6.0:
        soft_labels[0] = 1.0
    elif label > 42.0:
        soft_labels[num_classes-1] = 1.0
    return soft_labels

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patchlist_csv, slideid_to_survtime):
        df = pd.read_csv(patchlist_csv)
        self.items = []
        for _, row in df.iterrows():
            patch_path = row['path']
            # 親ディレクトリ名（患者ID）を取得
            patient_id = Path(patch_path).parent.name
            self.items.append((patch_path, patient_id))
        self.slideid_to_survtime = slideid_to_survtime

        # 画像変換
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        patch_path, patient_id = self.items[idx]
        img = Image.open(patch_path).convert('RGB')
        img = self.transform(img)
        survtime = torch.tensor(self.slideid_to_survtime[patient_id], dtype=torch.float)
        softlabel = create_softlabel_survival_time_wise(survtime)
        # 12ヶ月ごと4クラス
        if survtime < 12:
            y_class = 0
        elif survtime < 24:
            y_class = 1
        elif survtime < 36:
            y_class = 2
        else:
            y_class = 3
        return img, softlabel, survtime, patient_id

def load_split(split_path):
    df = pd.read_csv(split_path)
    train_ids = df['train'].dropna().astype(str).tolist()
    val_ids = df['val'].dropna().astype(str).tolist()
    test_ids = df['test'].dropna().astype(str).tolist()
    return train_ids, val_ids, test_ids

def load_survival_time(xlsx_path):
    df = pd.read_excel(xlsx_path)
    slideid_to_survtime = {}
    for _, row in df.iterrows():
        slideid = str(row['slide_id'])
        survtime = float(row['survival time'])
        slideid_to_survtime[slideid] = survtime
    return slideid_to_survtime

# クラスごとの代表値（中央値）
class_means = torch.tensor([6, 18, 30, 42], dtype=torch.float).to(device)

# モデルのロード
def load_model(model_path):
    model = AutoEncoder2()
    model.dec = torch.nn.Sequential(
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# テスト結果を保存する関数
def save_results(test_loader, model, save_dir):
    patient_predictions = defaultdict(list)
    patient_true_labels = {}
    with torch.no_grad():
        for x, _, y_true, subjects in tqdm(test_loader, desc="Test"):
            x = x.to(device)
            y_true = y_true.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=-1)
            pred = (probs * class_means).sum(dim=1)
            for i, subject in enumerate(subjects):
                # subjectがTensor型ならitem()でIDに変換
                sid = str(subject) if isinstance(subject, str) else str(subject.item()) if hasattr(subject, "item") else str(subject)
                patient_predictions[sid].append(pred[i].item())
                if sid not in patient_true_labels:
                    patient_true_labels[sid] = y_true[i].item()

    # スライド単位で生存期間値を平均
    patient_avg_predictions = {subject: np.mean(preds) for subject, preds in patient_predictions.items()}

    # C-index計算
    true_labels = [patient_true_labels[subject] for subject in patient_avg_predictions.keys()]
    predicted_labels = [patient_avg_predictions[subject] for subject in patient_avg_predictions.keys()]
    if len(set(true_labels)) > 1 and len(set(predicted_labels)) > 1:
        c_index = concordance_index(true_labels, predicted_labels)
    else:
        c_index = None
        print("No admissible pairs for C-index calculation.")

    # CSVに保存
    save_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame([
        {"slide_id": subject, "predicted_survival_time": predicted, "true_survival_time": patient_true_labels[subject]}
        for subject, predicted in patient_avg_predictions.items()
    ])
    results.to_csv(save_dir / "test_results.csv", index=False)
    with open(save_dir / "c_index.txt", "w") as f:
        if c_index is not None:
            f.write(f"C-index: {c_index:.4f}\n")
        else:
            f.write("C-index could not be calculated due to lack of admissible pairs.\n")
    print(f"Results saved to {save_dir / 'test_results.csv'}")
    print(f"C-index saved to {save_dir / 'c_index.txt'}")
    
def get_patient_id(slide_id):
    # "2-3" → "2"
    return slide_id.split('-')[0]

def main():
    import glob
    dataset_root = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology")
    splits_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/CLAM_rereclone/splits/mie_SARC_tumor_survival_4class_100")
    save_root = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_result/AE_CNN_model")
    survtime_xlsx = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/_data/survival_time_cls/AIpatho_20250424_subtype.xlsx")

    slideid_to_survtime = load_survival_time(survtime_xlsx)
    n_splits = 3  # fold0, fold1, fold2

    for split_idx in range(n_splits):
        print(f"\n=== Evaluating Fold {split_idx} ===")
        split_path = splits_dir / f"splits_{split_idx}.csv"
        _, _, test_ids = load_split(split_path)  # test_ids を取得
        print(f"Test IDs for Fold {split_idx}: {test_ids}")

        # テスト対象のスライドに対応するパッチリストを収集
        patchlist_paths = glob.glob(f"{dataset_root}/*/patchlist/patchlist_updated.csv")
        test_patch_items = []
        for patchlist_csv in patchlist_paths:
            df = pd.read_csv(patchlist_csv)
            slide_id = Path(patchlist_csv).parent.parent.name
            print(f"Processing Slide ID: {slide_id}, In Test IDs: {slide_id in test_ids}")
            if slide_id in test_ids:  # test_ids に含まれるスライドのみを対象
                df['slide_id'] = slide_id
                for _, row in df.iterrows():
                    patch_path = row['path']
                    test_patch_items.append((patch_path, slide_id))

        # DataLoader を作成
        class TestPatchDataset(torch.utils.data.Dataset):
            def __init__(self, patch_items, slideid_to_survtime):
                self.items = patch_items
                self.slideid_to_survtime = slideid_to_survtime
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                patch_path, slide_id = self.items[idx]
                img = Image.open(patch_path).convert('RGB')
                img = self.transform(img)
                survtime = torch.tensor(self.slideid_to_survtime[slide_id], dtype=torch.float)
                softlabel = create_softlabel_survival_time_wise(survtime)
                if survtime < 12:
                    y_class = 0
                elif survtime < 24:
                    y_class = 1
                elif survtime < 36:
                    y_class = 2
                else:
                    y_class = 3
                return img, softlabel, survtime, slide_id

        test_loader = torch.utils.data.DataLoader(
            TestPatchDataset(test_patch_items, slideid_to_survtime), batch_size=32, num_workers=4, drop_last=False
        )

        # モデルのロード
        model_path = Path(f"/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_model/AE_CNN_model/AE_CNN_3bunkatu_fold{split_idx}.pth")
        model = load_model(model_path)

        # 結果保存先
        save_dir = save_root / f"fold{split_idx}"
        save_results(test_loader, model, save_dir)

if __name__ == '__main__':
    main()