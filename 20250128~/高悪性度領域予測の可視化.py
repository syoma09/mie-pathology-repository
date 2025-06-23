import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def load_predictions(csv_path):
    return pd.read_csv(csv_path)

def reconstruct_image(predictions, output_path, scale_factor=0.1):
    # 画像のサイズを取得
    max_x = predictions['x'].max() + predictions['width'].max()
    max_y = predictions['y'].max() + predictions['height'].max()

    # 再構成する画像のサイズを計算
    img_width = int(max_x * scale_factor)
    img_height = int(max_y * scale_factor)

    # 空の画像を作成
    reconstructed_img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(reconstructed_img)

    for _, row in predictions.iterrows():
        x = int(row['x'] * scale_factor)
        y = int(row['y'] * scale_factor)
        width = int(row['width'] * scale_factor)
        height = int(row['height'] * scale_factor)
        path = row['path']
        predicted_severe = row['predicted_severe']

        # パッチ画像を読み込み
        if predicted_severe == 1:
            patch_img = Image.open(path).resize((width, height))
        else:
            patch_img = Image.new('RGB', (width, height), (255, 255, 255))

        # 高悪性度領域の場合、赤い枠を描画
        if predicted_severe == 1:
            draw.rectangle([x, y, x + width, y + height], outline='red', width=3)

        # パッチ画像を再構成画像に貼り付け
        reconstructed_img.paste(patch_img, (x, y))

    # 再構成した画像を保存
    reconstructed_img.save(output_path)

def process_all_csvs(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in input_dir.glob("*.csv"):
        predictions = load_predictions(csv_file)
        output_image_path = output_dir / (csv_file.stem + "_reconstructed.png")
        reconstruct_image(predictions, output_image_path)

def main():
    input_dir = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/predictions"
    output_dir = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/20250128~/reconstructed_images"

    # すべてのCSVファイルを処理して画像を再構成
    process_all_csvs(input_dir, output_dir)

if __name__ == '__main__':
    main()