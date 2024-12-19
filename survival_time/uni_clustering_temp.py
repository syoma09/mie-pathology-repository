#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# もとはuniencoder3.py

# まずは三重大学のデータでクラスタリング結果し。何色が腫瘍の位置を表しているかを確認したい　一枚だけ
# 画像に戻すためにはパッチ画像に座標を保持しておく必要性あり
import os
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import timm
import torch
from torchvision import transforms
from pathlib import Path
from huggingface_hub import login, hf_hub_download
import openslide
from aipatho.svs import TumorMasking, SVS
from aipatho.dataset import save_patches

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch
from torchvision import transforms
from pathlib import Path
from huggingface_hub import login, hf_hub_download
import openslide
import pandas as pd

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_patchlist(patchlist_path):
    patchlist = pd.read_csv(patchlist_path)
    return patchlist

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patchlist, transform):
        super(PatchDataset, self).__init__()
        self.transform = transform
        self.patchlist = patchlist

    def __len__(self):
        return len(self.patchlist)

    def __getitem__(self, item):
        row = self.patchlist.iloc[item]
        img_path = Path(row['path']).resolve()  # パスを正規化
        x, y = int(row['x']), int(row['y'])
        try:
            img = Image.open(img_path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {img_path}: {e}")
            return None  # Noneを返してエラーを回避
        
        img = self.transform(img)
        return img, (x, y)

def main():
    patchlist_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/51-4/patchlist/patchlist_updated.csv"
    patchlist = load_patchlist(patchlist_path)

    batch_size = 16
    num_workers = os.cpu_count() // 4

    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    login(token)

    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)
    model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True, pretrained=False
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(patchlist, transform), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    coordinates_list = []
    attention_scores = []

    with torch.no_grad():
        for batch, (x, coords) in enumerate(valid_loader):
            x = x.to(device)
            
            outputs = model(x)
            attentions = outputs[1]
            
            attention_scores.append(attentions.cpu())

            # coordsを(x, y)ペアに変換する
            x_coords = coords[0]  # x座標のテンソル
            y_coords = coords[1]  # y座標のテンソル
            
            for x_coord, y_coord in zip(x_coords, y_coords):  # xとyをペアにする
                coordinates_list.append((x_coord.item(), y_coord.item()))  # Tensorから数値に変換

            # 最初のバッチのみデバッグ表示
            if batch == 0:
                print(f"Coords Sample: {coordinates_list[:5]}")


    attention_scores_tensor = torch.cat(attention_scores, dim=0)

    # デバッグ用に一部の値を表示
    print("Coordinates List Sample:", coordinates_list[:5])
    print("Attention Scores Tensor Shape:", attention_scores_tensor.shape)
    print("Attention Scores Tensor Sample:", attention_scores_tensor[:5])

    slide = openslide.OpenSlide("/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/svs/51-4.svs")
    width, height = slide.dimensions

    # アテンションマップの解像度を縮小
    scale_factor = 0.1
    small_width, small_height = int(width * scale_factor), int(height * scale_factor)
    attention_map = torch.zeros((small_height, small_width), dtype=torch.float32, device=device)

    # 各パッチのアテンションスコアを計算し、縮小したアテンションマップに反映
    for (x, y), attention in zip(coordinates_list, attention_scores_tensor):
        small_x, small_y = int(x * scale_factor), int(y * scale_factor)
        patch_size = int(512 * scale_factor)
        attention_value = attention.mean().item()
        attention_map[small_y:small_y+patch_size, small_x:small_x+patch_size] = attention_value

    # アテンションマップをCPUで行う
    attention_map = attention_map.cpu().numpy()

    # 元のスライド画像を読み込む
    slide_image = slide.read_region((0, 0), 0, (width, height)).convert('RGB')

    # スライド画像のサイズを確認
    print(f"Slide image size: {slide_image.size}")

    # アテンションマップをカラー化
    attention_map_colored = np.zeros((attention_map.shape[0], attention_map.shape[1], 3), dtype=np.uint8)

    # マイナスの値を青にする
    attention_map_colored[attention_map < 0] = [0, 0, 255]

    # 0以上の値を青から赤のグラデーションで表現
    positive_attention_map = attention_map >= 0
    normalized_attention = attention_map[positive_attention_map] / attention_map[positive_attention_map].max()
    colored_attention = plt.cm.jet(normalized_attention)[:, :3] * 255
    attention_map_colored[positive_attention_map] = colored_attention.astype(np.uint8)
    
    attention_map_image = Image.fromarray(attention_map_colored).convert('RGB')
    attention_map_image = attention_map_image.resize(slide_image.size, Image.Resampling.LANCZOS)

    # アテンションマップ画像のサイズを確認
    print(f"Attention map image size: {attention_map_image.size}")

    # 元のスライド画像とアテンションマップを重ねる
    combined_image = Image.blend(slide_image, attention_map_image, alpha=0.5)
    # 重ねた画像を縮小して保存
    scale_factor = 0.01
    small_width, small_height = int(combined_image.width * scale_factor), int(combined_image.height * scale_factor)
    combined_image_resized = combined_image.resize((small_width, small_height), Image.Resampling.LANCZOS)
    combined_image_resized.save("/net/nfs3/export/home/sakakibara/data/51-4_zahyoutuki_patch/attention_map_resized.png")


if __name__ == '__main__':
    main()