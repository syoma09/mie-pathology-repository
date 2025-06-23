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
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# デバイス設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# パッチリストのロード関数
def load_patchlist(patchlist_path):
    return pd.read_csv(patchlist_path)

# モデルのロード関数
def load_model():
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
    return model

# 前処理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# アテンションスコアの取得関数
def calculate_attention_scores(model, patchlist, transform, device):
    attention_scores = []
    for _, row in tqdm(patchlist.iterrows(), total=len(patchlist), desc="Calculating attention scores"):
        img_path = row['path']
        x, y = row['x'], row['y']
        
        # パッチ画像の読み込みと前処理
        patch_image = Image.open(img_path).convert("RGB")
        input_tensor = transform(patch_image).unsqueeze(0).to(device)
        
        # モデル推論とアテンションスコア取得
        with torch.no_grad():
            outputs = model.forward_features(input_tensor)
            cls_attention = outputs[:, 0, :]  # CLSトークンのアテンションスコア
            attention_scores.append((x, y, cls_attention.mean().item()))
    
    return attention_scores

# ヒートマップの作成
def create_heatmap(patchlist, attention_scores, original_width, original_height):
    heatmap = np.zeros((original_height, original_width))
    mask = np.zeros((original_height, original_width), dtype=bool)
    
    for x, y, score in attention_scores:
        heatmap[y:y+512, x:x+512] = score
        mask[y:y+512, x:x+512] = True
    
    # 正規化
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap, mask

# ヒートマップの保存
def save_heatmap(heatmap, mask, slide, output_path, scale_factor=0.1, cmap='inferno'):
    resized_width, resized_height = int(slide.dimensions[0] * scale_factor), int(slide.dimensions[1] * scale_factor)
    slide_thumbnail = slide.get_thumbnail((resized_width, resized_height))

    resized_heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    resized_heatmap = resized_heatmap.resize((resized_width, resized_height), resample=Image.BILINEAR)
    
    colored_heatmap = plt.cm.get_cmap(cmap)((np.array(resized_heatmap) / 255.0))[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    colored_heatmap = Image.fromarray(colored_heatmap)

    if slide_thumbnail.size != colored_heatmap.size:
        colored_heatmap = colored_heatmap.resize(slide_thumbnail.size, resample=Image.BILINEAR)

    # マスクを適用して、パッチ画像がない部分は元の画像の色のままにする
    composite_image = Image.new("RGBA", slide_thumbnail.size)
    for y in range(slide_thumbnail.size[1]):
        for x in range(slide_thumbnail.size[0]):
            if mask[int(y / scale_factor), int(x / scale_factor)]:
                composite_image.putpixel((x, y), colored_heatmap.getpixel((x, y)))
            else:
                composite_image.putpixel((x, y), slide_thumbnail.getpixel((x, y)))

    composite_image.save(output_path)
    print(f"Heatmap saved to: {output_path}")

def process_svs_files(src_dir, dst_dir):
    # src_dir内のすべてのsvsファイルを取得
    svs_files = list(Path(src_dir).rglob("*.svs"))

    model = load_model()

    for svs_file in tqdm(svs_files, desc="Processing SVS files"):
        # svsファイルのファイル名（拡張子なし）を取得
        file_stem = svs_file.stem

        # 保存先ディレクトリを作成
        save_dir = Path(dst_dir) / file_stem
        save_dir.mkdir(parents=True, exist_ok=True)

        # attention.pngの保存先パスを設定
        save_path = save_dir / "attention.png"

        # スライド画像の読み込み
        slide = openslide.OpenSlide(str(svs_file))

        # パッチリストのパスとロード
        patchlist_path = f"/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/{file_stem}/patchlist/patchlist_updated.csv"
        if not os.path.exists(patchlist_path):
            print(f"Patchlist not found for {file_stem}, skipping...")
            continue
        patchlist = load_patchlist(patchlist_path)

        # アテンションスコアの計算
        attention_scores = calculate_attention_scores(model, patchlist, transform, device)

        # 元画像サイズに合わせたヒートマップ生成
        original_width, original_height = slide.dimensions
        heatmap, mask = create_heatmap(patchlist, attention_scores, original_width, original_height)

        # ヒートマップの保存
        save_heatmap(heatmap, mask, slide, save_path, scale_factor=0.1)

def update_patchlist_paths(patchlist_path, updated_patchlist_path):
    df = pd.read_csv(patchlist_path)
    df['path'] = df['path'].str.replace('/net/nfs2/', '/net/nfs3/')
    df.to_csv(updated_patchlist_path, index=False)
    print(f"Updated patchlist saved to: {updated_patchlist_path}")

def main():
    src_dir = "/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"
    dst_dir = "/net/nfs3/export/home/sakakibara/data/attention_heatmap/"

    process_svs_files(src_dir, dst_dir)

if __name__ == '__main__':
    main()