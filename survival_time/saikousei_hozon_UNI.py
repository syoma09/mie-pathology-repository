import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from aipatho.svs import TumorMasking
from aipatho.model import AutoEncoder2
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import PatchDataset, load_annotation
from aipatho.metrics.label import TimeToTime

# デバイスの選択
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def save_images(input_images, reconstructed_images, save_dir, prefix='img'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (input_img, recon_img) in enumerate(zip(input_images, reconstructed_images)):
        input_img = transforms.ToPILImage()(input_img.cpu())
        recon_img = (recon_img + 1) / 2 * 255  # [-1, 1]を[0, 255]に変換
        recon_img = recon_img.clip(0, 255).to(torch.uint8)
        recon_img = transforms.ToPILImage()(recon_img.cpu())
        input_img.save(os.path.join(save_dir, f"{prefix}_input_{i}.png"))
        recon_img.save(os.path.join(save_dir, f"{prefix}_recon_{i}.png"))

# UNIモデルのViT部分だけを使う
def load_uni_feature_extractor():
    import timm
    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    base_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    base_model.eval()
    base_model.to(device)
    return base_model

# UNI特徴量(1024)→AE特徴量(512)への変換層
class UNI2AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 512)
    def forward(self, x):
        x = self.fc(x)  # [B, 512]
        x = x.view(x.size(0), 512, 1, 1)  # [B, 512, 1, 1]
        return x

def main():
    # AEのデコーダのみ利用
    model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20250616_230800/state00014.pth"
    ae = AutoEncoder2().to(device)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()
    decoder = ae.dec

    # UNI特徴抽出器
    uni = load_uni_feature_extractor()

    # UNI→AE特徴量変換層（※学習済みでない場合はランダム初期化なので画像は崩れます）
    uni2ae = UNI2AE().to(device)
    # uni2ae.load_state_dict(torch.load("uni2ae.pth"))  # 学習済みがあれば

    # データセットの準備
    patch_size = 224, 224  # ViTの入力サイズに合わせる
    stride = 224, 224
    target = TumorMasking.FULL

    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=target
    )

    annotation_path = Path("_data/survival_time_cls/20220413_aut2.csv")
    annotation = load_annotation(annotation_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # ViTの前処理に合わせる場合は正規化も追加
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_dataset = PatchDataset(dataset_root, annotation['valid'], transform=transform, labeler=TimeToTime())
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=5,
        shuffle=True
    )

    save_dir = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/saikousei_uni"
    with torch.no_grad():
        for batch, (x, _) in enumerate(valid_loader):
            x = x.to(device)
            # UNI特徴量抽出
            features_uni = uni(x)  # [B, 1024]
            # UNI→AE特徴量
            features_ae = uni2ae(features_uni)
            # AEデコーダで再構成
            recon_img = decoder(features_ae)
            #recon_img = recon_img.view(-1, 3, patch_size[0], patch_size[1])
            save_images(x, recon_img, save_dir, prefix=f"batch{batch}")
            break  # 1バッチのみ処理

if __name__ == '__main__':
    main()