#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# もとはuniencoder3.py

# まずは三重大学のデータでクラスタリング結果し。何色が腫瘍の位置を表しているかを確認したい　一枚だけ
# 画像に戻すためにはパッチ画像に座標を保持しておく必要性あり
#gridrinのchatgpt ViT-L/16を使って、Attention Score を参照
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
import timm
import openslide
from PIL import Image

# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# パッチリストのロード関数
def load_patchlist(patchlist_path):
    return pd.read_csv(patchlist_path)

# パッチリストのパスとロード
patchlist_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/51-4/patchlist/patchlist_updated.csv"
patchlist = load_patchlist(patchlist_path)

# バッチサイズとワーカー数
batch_size = 16
num_workers = os.cpu_count() // 4

# Hugging Face Hubのトークンでログイン
token = os.getenv('HUGGINGFACE_HUB_TOKEN')
login(token)

# モデルファイルのダウンロードと確認
local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
os.makedirs(local_dir, exist_ok=True)
model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)

if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

# モデルのロード
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True, pretrained=False
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
model.eval()
model.to(device)

# 前処理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# スライド画像の読み込み
slide_path = "/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/svs/51-4.svs"
slide = openslide.OpenSlide(slide_path)

# アテンションスコアの取得関数
def calculate_attention_scores(model, patchlist, transform, device):
    attention_scores = []
    for _, row in patchlist.iterrows():
        img_path = row['path']
        x, y = row['x'], row['y']
        
        # パッチ画像の読み込みと前処理
        patch_image = Image.open(img_path).convert("RGB")
        input_tensor = transform(patch_image).unsqueeze(0).to(device)
        
        # モデル推論とアテンションスコア取得
        with torch.no_grad():
            outputs = model.forward_features(input_tensor)
            cls_attention = outputs[:, 0, :]  # CLSトークンのアテンションスコア
            attention_scores.append(cls_attention.mean().item())
    
    return attention_scores

# アテンションスコアの計算
attention_scores = calculate_attention_scores(model, patchlist, transform, device)

# ヒートマップの作成
def create_heatmap(patchlist, attention_scores, original_width, original_height):
    heatmap = np.zeros((original_height, original_width))
    
    for (x, y), score in zip(patchlist[['x', 'y']].values, attention_scores):
        heatmap[y:y+512, x:x+512] = score #本来256だが、512に変更することでヒートマップに隙間がなくなる
    
    # 正規化
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap

# 元画像サイズに合わせたヒートマップ生成
original_width, original_height = slide.dimensions
heatmap = create_heatmap(patchlist, attention_scores, original_width, original_height)

# ヒートマップの保存
def save_heatmap(heatmap, slide, output_path, scale_factor=0.1, cmap='inferno'):
    """
    ヒートマップを元画像に重ね合わせて保存。
    :param heatmap: ヒートマップ (numpy array)
    :param slide: OpenSlideオブジェクト
    :param output_path: 保存先のファイルパス
    :param scale_factor: 縮小率（0.1なら10分の1）
    :param cmap: カラーマップ
    """
    # 元画像の縮小
    resized_width, resized_height = int(slide.dimensions[0] * scale_factor), int(slide.dimensions[1] * scale_factor)
    slide_thumbnail = slide.get_thumbnail((resized_width, resized_height))

    # ヒートマップの縮小
    resized_heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    resized_heatmap = resized_heatmap.resize((resized_width, resized_height), resample=Image.BILINEAR)
    
    # カラーマップの適用
    colored_heatmap = plt.colormaps[cmap]((np.array(resized_heatmap) / 255.0))[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    colored_heatmap = Image.fromarray(colored_heatmap)

    # 元画像とヒートマップの合成
    composite_image = Image.blend(slide_thumbnail.convert("RGBA"), colored_heatmap.convert("RGBA"), alpha=0.5)

    # 保存
    composite_image.save(output_path)
    print(f"Heatmap saved to: {output_path}")

# 保存先パス
output_path = "/net/nfs3/export/home/sakakibara/data/51-4_zahyoutuki_patch/attention_gomi.png"
save_heatmap(heatmap, slide, output_path, scale_factor=0.1)  # 縮小率を変更可能
