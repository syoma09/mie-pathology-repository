import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class PatchDataset(Dataset):
    def __init__(self, root, annotations):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = []
        self.annotations = annotations

        for subject in annotations:
            patchlist_path = root / subject / 'patchlist' / 'patchlist_severe.csv'
            if patchlist_path.exists():
                with open(patchlist_path, 'r') as f:
                    next(f)  # Skip header
                    for line in f:
                        x, y, width, height, path, severe = line.strip().split(',')
                        self.dataset.append((x, y, width, height, path, int(severe), subject))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y, width, height, path, severe, subject = self.dataset[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, severe, subject, x, y, width, height, path

class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(1024, 512)
        self.additional_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1, bias=True)
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.base_model(x)
        features = self.fc(features)
        output = self.additional_layers(features)
        return output

def load_model(model_path):
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model = CustomModel(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, output_dir):
    all_preds = []
    all_labels = []
    all_subjects = []
    all_x = []
    all_y = []
    all_width = []
    all_height = []
    all_paths = []
    with torch.no_grad():
        for inputs, labels, subjects, x, y, width, height, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.round(torch.sigmoid(outputs.squeeze()))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(subjects)
            all_x.extend(x)
            all_y.extend(y)
            all_width.extend(width)
            all_height.extend(height)
            all_paths.extend(paths)

    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds)
    overall_recall = recall_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds)

    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    print(f'Overall Precision: {overall_precision:.4f}')
    print(f'Overall Recall: {overall_recall:.4f}')
    print(f'Overall F1 Score: {overall_f1:.4f}')

    # 各numberごとの精度を計算
    subject_metrics = {}
    for subject in set(all_subjects):
        subject_indices = [i for i, s in enumerate(all_subjects) if s == subject]
        subject_labels = [all_labels[i] for i in subject_indices]
        subject_preds = [all_preds[i] for i in subject_indices]

        accuracy = accuracy_score(subject_labels, subject_preds)
        precision = precision_score(subject_labels, subject_preds)
        recall = recall_score(subject_labels, subject_preds)
        f1 = f1_score(subject_labels, subject_preds)

        subject_metrics[subject] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # 各subjectごとに予測結果をCSVに保存
        subject_data = {
            'x': [all_x[i] for i in subject_indices],
            'y': [all_y[i] for i in subject_indices],
            'width': [all_width[i] for i in subject_indices],
            'height': [all_height[i] for i in subject_indices],
            'path': [all_paths[i] for i in subject_indices],
            'severe': [all_labels[i] for i in subject_indices],
            'predicted_severe': [all_preds[i] for i in subject_indices]
        }
        subject_df = pd.DataFrame(subject_data)
        subject_df.to_csv(output_dir / f'{subject}_predictions.csv', index=False)

    for subject, metrics in subject_metrics.items():
        print(f'Subject {subject} - Accuracy: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1 Score: {metrics["f1"]:.4f}')

def main():
    model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20250317_212906uniencoder3/model00215.pth"
    csv_path = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/_data/survival_time_cls/20220726_cls.csv"
    dataset_root = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/")
    output_dir = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSVファイルを読み込み、tvt=1の行をフィルタリング
    df = pd.read_csv(csv_path)
    annotations = df[df['tvt'] == 1]['number'].tolist()

    # データセットとデータローダーの作成
    dataset = PatchDataset(dataset_root, annotations)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count() // 4)

    # モデルのロード
    model = load_model(model_path)

    # モデルの評価
    evaluate_model(model, dataloader, output_dir)

if __name__ == '__main__':
    main()