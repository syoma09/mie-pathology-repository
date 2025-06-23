# 3分割で生存期間分類学習するコード(20250616)(mie_pathology全データ入ってなかったから、全データ学習しきれてない)
import numpy as np
import os
import datetime
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import transforms
from PIL import Image
from lifelines.utils import concordance_index
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from aipatho.model.autoencoder2 import AutoEncoder2
from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.metrics import MeanVarianceLoss

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
    def __init__(self, root, slide_ids, slideid_to_survtime):
        super(PatchDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []
        for subject in slide_ids:
            slide_dir = root / subject
            if not slide_dir.exists():
                continue
            for path in slide_dir.iterdir():
                self.__dataset.append((path, subject))
        random.shuffle(self.__dataset)
        self.slideid_to_survtime = slideid_to_survtime

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, subject = self.__dataset[item]
        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            return self.__getitem__((item + 1) % len(self.__dataset))
        img = self.transform(img)
        survtime = torch.tensor(self.slideid_to_survtime[subject], dtype=torch.float)
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
        return img, softlabel, survtime, y_class

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

def main():
    import argparse
    patch_size = 256, 256
    stride = 256, 256

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )

    # 3分割用splitファイル
    splits_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/CLAM_rereclone/splits/mie_SARC_tumor_survival_4class_100")
    n_splits = 3  # split_0.csv, split_1.csv, split_2.csv

    epochs = 30
    batch_size = 32
    num_workers = os.cpu_count() // 4

    survtime_xlsx = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/_data/survival_time_cls/AIpatho_20250424_subtype.xlsx")
    slideid_to_survtime = load_survival_time(survtime_xlsx)

    # モデル・ログ保存先
    save_root = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_model/AE_CNN_model")
    save_root.mkdir(parents=True, exist_ok=True)

    # AEモデルの準備
    model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20250616_230800/state00014.pth"
    base_model = AutoEncoder2()
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    base_model.dec = nn.Sequential(
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
    base_model = base_model.to(device)
    base_model.eval()

    # クラスごとの代表値（中央値）
    class_means = torch.tensor([6, 18, 30, 42], dtype=torch.float).to(device)

    # 損失関数
    LAMBDA_1 = 0.2
    LAMBDA_2 = 0.05
    START_AGE = 0
    END_AGE = 3
    criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE)
    criterion2 = nn.KLDivLoss(reduction='batchmean')

    # argparseでfold指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None, help='fold index (0, 1, 2)')
    args = parser.parse_args()
    if args.fold is not None:
        split_indices = [args.fold]
    else:
        split_indices = range(n_splits)

    for split_idx in split_indices:
        print(f"\n=== Split {split_idx} ===")
        split_path = splits_dir / f"splits_{split_idx}.csv"
        train_ids, val_ids, test_ids = load_split(split_path)

        train_loader = torch.utils.data.DataLoader(
            PatchDataset(dataset_root, train_ids, slideid_to_survtime), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )
        valid_loader = torch.utils.data.DataLoader(
            PatchDataset(dataset_root, val_ids, slideid_to_survtime), batch_size=batch_size,
            num_workers=num_workers, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            PatchDataset(dataset_root, test_ids, slideid_to_survtime), batch_size=batch_size,
            num_workers=num_workers, drop_last=True
        )

        model = base_model
        optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        # ログ保存先
        log_root = save_root / f"AE_CNN_3bunkatu_log_fold{split_idx}"
        log_root.mkdir(parents=True, exist_ok=True)
        tensorboard = SummaryWriter(log_dir=log_root, filename_suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

        for epoch in range(epochs):
            print(f"Fold {split_idx} Epoch [{epoch+1}/{epochs}]")
            model.train()
            train_loss = 0.
            train_mean_loss = 0.
            train_variance_loss = 0.
            train_softmax_loss = 0.
            train_mae = 0.
            train_index = 0.
            for batch, (x, soft_labels, y_true, y_class) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
                optimizer.zero_grad()
                x = x.to(device)
                soft_labels = soft_labels.to(device)
                y_true = y_true.to(device)
                y_class = y_class.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    mean_loss, variance_loss = criterion1(outputs, y_class, device)
                    softmax_loss = criterion2(torch.log_softmax(outputs, dim=1), soft_labels)
                    loss = mean_loss + variance_loss + softmax_loss
                    if torch.isnan(loss):
                        print("Warning: NaN detected in total loss.")
                        continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                probs = torch.softmax(outputs, dim=-1)
                pred = (probs * class_means).sum(dim=1)
                mae = torch.abs(pred - y_true).mean().item()
                status = np.ones(len(y_true))
                index = concordance_index(y_true.cpu().numpy(), pred.detach().cpu().numpy(), status)
                train_loss += loss.item() / len(train_loader)
                train_mean_loss += mean_loss / len(train_loader)
                train_variance_loss += variance_loss / len(train_loader)
                train_softmax_loss += softmax_loss / len(train_loader)
                train_mae += mae / len(train_loader)
                train_index += index / len(train_loader)
            print(f"\n[Fold {split_idx}] Epoch {epoch+1} Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} CI: {train_index:.4f}")

            # バリデーション
            model.eval()
            valid_loss = 0.
            valid_mean_loss = 0.
            valid_variance_loss = 0.
            valid_softmax_loss = 0.
            valid_mae = 0.
            valid_index = 0.
            with torch.no_grad():
                for x, soft_labels, y_true, y_class in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}"):
                    x = x.to(device)
                    soft_labels = soft_labels.to(device)
                    y_true = y_true.to(device)
                    y_class = y_class.to(device)
                    outputs = model(x)
                    mean_loss, variance_loss = criterion1(outputs, y_class, device)
                    softmax_loss = criterion2(torch.log_softmax(outputs, dim=1), soft_labels)
                    loss = mean_loss + variance_loss + softmax_loss
                    probs = torch.softmax(outputs, dim=-1)
                    pred = (probs * class_means).sum(dim=1)
                    mae = torch.abs(pred - y_true).mean().item()
                    status = np.ones(len(y_true))
                    index = concordance_index(y_true.cpu().numpy(), pred.detach().cpu().numpy(), status)
                    valid_loss += loss.item() / len(valid_loader)
                    valid_mean_loss += mean_loss / len(valid_loader)
                    valid_variance_loss += variance_loss / len(valid_loader)
                    valid_softmax_loss += softmax_loss / len(valid_loader)
                    valid_mae += mae / len(valid_loader)
                    valid_index += index / len(valid_loader)
            print(f"[Fold {split_idx}] Epoch {epoch+1} Valid Loss: {valid_loss:.4f} MAE: {valid_mae:.4f} CI: {valid_index:.4f}")

            # TensorBoardへの書き込み（UNI_CNN_3bunkatu.pyと同じ形式）
            tensorboard.add_scalar('train_MV', train_loss, epoch)
            tensorboard.add_scalar('train_Mean', train_mean_loss, epoch)
            tensorboard.add_scalar('train_Variance', train_variance_loss, epoch)
            tensorboard.add_scalar('train_Softmax', train_softmax_loss, epoch)
            tensorboard.add_scalar('train_MAE', train_mae, epoch)
            tensorboard.add_scalar('train_Index', train_index, epoch)
            tensorboard.add_scalar('valid_MV', valid_loss, epoch)
            tensorboard.add_scalar('valid_Mean', valid_mean_loss, epoch)
            tensorboard.add_scalar('valid_Variance', valid_variance_loss, epoch)
            tensorboard.add_scalar('valid_Softmax', valid_softmax_loss, epoch)
            tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
            tensorboard.add_scalar('valid_Index', valid_index, epoch)

        # モデル保存（30epoch後のみ）
        model_save_path = save_root / f"AE_CNN_3bunkatu_fold{split_idx}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model to {model_save_path}")

        # テスト（患者ごと平均予測）
        model.eval()
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
                    patient_predictions[subject].append(pred[i].item())
                    if subject not in patient_true_labels:
                        patient_true_labels[subject] = y_true[i].item()
        patient_avg_predictions = {subject: np.mean(preds) for subject, preds in patient_predictions.items()}
        print(f"\n[Fold {split_idx}] test 患者ごとの予測結果:")
        for subject in patient_avg_predictions:
            predicted = patient_avg_predictions[subject]
            actual = patient_true_labels[subject]
            print(f"患者 {subject}: 予測生存期間 = {predicted:.2f}, 実際の生存期間 = {actual:.2f}")

if __name__ == '__main__':
    main()