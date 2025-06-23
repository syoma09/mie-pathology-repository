#2025/06/22 AEのテスト画像の生存期間予測値をヒートマップ化→全患者4クラス分類されているから最小値最大値は大体6~38
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pathlib import Path
from torchvision import transforms
from aipatho.model.autoencoder2 import AutoEncoder2
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

def get_patch_color_with_colormap(predicted_survival_time, min_survival=0, max_survival=48, colormap='inferno'):
    """
    生存期間予測値に基づいてカラーマップを使用して色を決定
    """
    norm = Normalize(vmin=min_survival, vmax=max_survival)
    cmap = colormaps[colormap]
    rgba = cmap(norm(predicted_survival_time))
    return tuple(int(255 * c) for c in rgba[:3])

def reconstruct_slide(patch_data, model, min_survival=0, max_survival=48, colormap='inferno'):
    """
    パッチデータを元にスライド画像を再構成し、各パッチの予測値リストも返す
    """
    max_x = int(patch_data["x"].max()) + int(patch_data["width"].max())
    max_y = int(patch_data["y"].max()) + int(patch_data["height"].max())

    slide_image = Image.new("RGB", (max_x, max_y), (255, 255, 255))
    draw = ImageDraw.Draw(slide_image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    predicted_list = []

    for _, row in patch_data.iterrows():
        x, y, width, height, path = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"]), row["path"]
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=-1)
            class_means = torch.tensor([6, 18, 30, 42], dtype=torch.float).to(device)
            predicted_survival_time = (probs * class_means).sum(dim=1).item()
        predicted_list.append(predicted_survival_time)
        color = get_patch_color_with_colormap(predicted_survival_time, min_survival, max_survival, colormap)
        draw.rectangle([x, y, x + width, y + height], fill=color)

    return slide_image, predicted_list

def main():
    # 設定
    dataset_root = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology")
    splits_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/CLAM_rereclone/splits/mie_SARC_tumor_survival_4class_100")
    output_base_dir = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_highrisk_heatmap/AE_CNN_model/")
    model_base_path = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/save_model/AE_CNN_model/")

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
        model_path = model_base_path / f"AE_CNN_3bunkatu_fold{split_idx}.pth"
        model = load_model(str(model_path))

        # patchlist探索
        patchlist_paths = glob.glob(str(dataset_root / "*/patchlist/patchlist_updated.csv"))
        slide_patchlist = {Path(p).parent.parent.name: p for p in patchlist_paths}

        total = len(test_ids)
        processed = 0

        for slide_id in test_ids:
            if slide_id not in slide_patchlist:
                print(f"[{processed+1}/{total}] {slide_id}: patchlistが見つかりません。スキップします。")
                continue

            coordinates_file = slide_patchlist[slide_id]
            output_dir = output_base_dir / f"fold{split_idx}" / slide_id
            output_path = output_dir / "reconstructed_slide_resized.png"
            os.makedirs(output_dir, exist_ok=True)

            patch_data = load_patch_coordinates(coordinates_file)
            slide_image, predicted_list = reconstruct_slide(patch_data, model, min_survival=0, max_survival=48, colormap='inferno')
            resized_slide_image = slide_image.resize(
                (slide_image.width // 10, slide_image.height // 10), Image.Resampling.LANCZOS
            )
            resized_slide_image.save(output_path)
            processed += 1
            print(f"[{processed}/{total}] {slide_id}: 再構成したスライド画像を保存しました: {output_path}")
            print(f"  └ パッチ予測値 min={min(predicted_list):.2f}, max={max(predicted_list):.2f}")

        print(f"Fold{split_idx}: 全{processed}件のスライドの処理が完了しました。")

if __name__ == "__main__":
    main()