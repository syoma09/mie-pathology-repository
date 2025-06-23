import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    import timm
    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    base_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    base_model.eval()
    base_model.to(device)

    class CustomModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.fc = torch.nn.Linear(1024, 512)
            self.additional_layers = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(512, 512, bias=True),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 512, bias=True),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 4, bias=True),  # 4クラス
            )
            for param in self.base_model.parameters():
                param.requires_grad = False

        def forward(self, x):
            features = self.base_model(x)
            features = self.fc(features)
            output = self.additional_layers(features)
            return output

    model = CustomModel(base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_patch_coordinates(file_path):
    """
    座標データを読み込む関数
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "width", "height", "path"], skiprows=1)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    return df

def reconstruct_patch_predictions(patch_data, model, min_survival=0, max_survival=48):
    """
    各パッチの予測生存期間リストを返す
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    predicted_list = []
    for _, row in patch_data.iterrows():
        path = row["path"]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=-1)
            class_means = torch.tensor([6, 18, 30, 42], dtype=torch.float).to(device)
            predicted_survival_time = (probs * class_means).sum(dim=1).item()
        predicted_list.append(predicted_survival_time)
        


def reconstruct_slide(patch_data, model, min_survival=0, max_survival=48, colormap='inferno'):
    """
    パッチデータを元に各パッチの予測値リストのみ返す
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    predicted_list = []
    for _, row in patch_data.iterrows():
        path = row["path"]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=-1)
            class_means = torch.tensor([6, 18, 30, 42], dtype=torch.float).to(device)
            predicted_survival_time = (probs * class_means).sum(dim=1).item()
        predicted_list.append(predicted_survival_time)
    return None, predicted_list

def main():
    import glob
    from pathlib import Path

    # 設定
    dataset_root = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology")
    splits_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/CLAM_rereclone/splits/mie_SARC_tumor_survival_4class_100")
    output_base_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_highrisk_heatmap/UNI_CNN_model/")
    model_base_path = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_model/UNI_CNN_model/")

    n_splits = 3  # fold0, fold1, fold2

    for split_idx in range(n_splits):
        print(f"\n=== Fold {split_idx} の処理を開始 ===")
        split_path = splits_dir / f"splits_{split_idx}.csv"
        if not split_path.exists():
            print(f"{split_path} が見つかりません。スキップします。")
            continue

        # test_idsの取得
        df_split = pd.read_csv(split_path)
        test_ids = df_split['test'].dropna().astype(str).tolist()
        print(f"Fold{split_idx}のテストスライド: {test_ids}")

        # モデルロード
        model_path = model_base_path / f"UNI_CNN_3bunkatu_fold{split_idx}.pth"
        model = load_model(str(model_path))

        # patchlist探索
        patchlist_paths = glob.glob(str(dataset_root / "*/patchlist/patchlist_updated.csv"))
        slide_patchlist = {Path(p).parent.parent.name: p for p in patchlist_paths}

        # バイオリンプロット用データ
        all_patch_preds = []
        all_slide_ids = []

        for slide_id in test_ids:
            if slide_id not in slide_patchlist:
                print(f"{slide_id}: patchlistが見つかりません。スキップします。")
                continue

            coordinates_file = slide_patchlist[slide_id]
            patch_data = load_patch_coordinates(coordinates_file)
            # 各パッチの予測値のみ取得
            _, predicted_list = reconstruct_slide(patch_data, model, min_survival=0, max_survival=48, colormap='inferno')
            all_patch_preds.extend(predicted_list)
            all_slide_ids.extend([slide_id] * len(predicted_list))

        # バイオリンプロット
        if all_patch_preds:
            plt.figure(figsize=(max(10, len(test_ids)//2), 6))
            df = pd.DataFrame({'slide_id': all_slide_ids, 'pred_survival': all_patch_preds})
            sns.violinplot(x='slide_id', y='pred_survival', data=df, scale='count', inner='box')
            plt.xticks(rotation=90)
            plt.ylabel('Predicted Survival Time (months)')
            plt.xlabel('Slide ID')
            plt.title(f'Violin plot of predicted survival time per patch (Fold {split_idx})')
            plt.tight_layout()
            violin_path = output_base_dir / f"fold{split_idx}_violinplot.png"
            plt.savefig(violin_path)
            plt.close()
            print(f"バイオリンプロットを保存しました: {violin_path}")

if __name__ == "__main__":
    main()