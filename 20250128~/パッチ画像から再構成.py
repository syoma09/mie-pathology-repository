#パッチ画像から病理画像を再構成するスクリプト。一枚だけ
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def reconstruct_image(patient_number, src_dir, patchlist_dir, output_dir, scale_factor=0.1):
    patchlist_path = Path(patchlist_dir) / patient_number / "patchlist" / "patchlist_updated.csv"
    if not patchlist_path.exists():
        print(f"Skipping {patient_number}: patchlist file not found.")
        return

    patchlist = pd.read_csv(patchlist_path)
    patch_images = []

    # パッチ画像の読み込みと再構成
    for _, row in tqdm(patchlist.iterrows(), total=len(patchlist), desc="Reconstructing image"):
        patch_path = row["path"]
        x, y, width, height = row["x"], row["y"], row["width"], row["height"]
        patch_image = Image.open(patch_path).convert("RGB")
        patch_images.append((patch_image, x, y, width, height))

    # 再構成する画像のサイズを計算
    max_x = max(row["x"] + row["width"] for _, row in patchlist.iterrows())
    max_y = max(row["y"] + row["height"] for _, row in patchlist.iterrows())
    reconstructed_image = Image.new("RGB", (max_x, max_y))

    # パッチ画像を再構成
    for patch_image, x, y, width, height in patch_images:
        reconstructed_image.paste(patch_image, (x, y))

    # 画像を縮小
    resized_image = reconstructed_image.resize(
        (int(reconstructed_image.width * scale_factor), int(reconstructed_image.height * scale_factor)),
        Image.Resampling.LANCZOS
    )

    # 保存先ディレクトリを作成
    output_path = Path(output_dir) / f"{patient_number}_reconstructed.png"
    os.makedirs(output_dir, exist_ok=True)

    # 画像を保存
    resized_image.save(output_path)
    print(f"Reconstructed image saved to {output_path}")

def main():
    patient_number = "122-9"
    src_dir = "/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12"
    patchlist_dir = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology"
    output_dir = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/20250128~"

    reconstruct_image(patient_number, src_dir, patchlist_dir, output_dir)

if __name__ == "__main__":
    main()