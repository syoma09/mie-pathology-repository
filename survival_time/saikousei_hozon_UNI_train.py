import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from aipatho.svs import TumorMasking
from aipatho.model import AutoEncoder2
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import PatchDataset, load_annotation
from aipatho.metrics.label import TimeToTime

# デバイスの選択
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

# UNI特徴量(1024)→AE特徴量(512,1,1)への変換層
class UNI2AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 512)
    def forward(self, x):
        x = self.fc(x)  # [B, 512]
        x = x.view(x.size(0), 512, 1, 1)  # [B, 512, 1, 1]
        return x

def main():
    # AEモデルの準備
    model_path = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/20250616_230800/state00014.pth"
    ae = AutoEncoder2().to(device)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()  # AEは固定

    # UNI特徴抽出器
    uni = load_uni_feature_extractor()

    # UNI→AE特徴量変換層（学習対象）
    uni2ae = UNI2AE().to(device)

    # データセットの準備
    patch_size = 224, 224
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
    ])

    train_dataset = PatchDataset(dataset_root, annotation['train'], transform=transform, labeler=TimeToTime())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    # 最適化対象はuni2aeのみ
    optimizer = optim.Adam(uni2ae.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 10
    save_dir = "./uni2ae_ckpt"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        uni2ae.train()
        total_loss = 0
        for batch, (x, _) in enumerate(train_loader):
            x = x.to(device)
            with torch.no_grad():
                # AEエンコーダで特徴量抽出
                z_ae = ae.encoder(x)  # [B, 512]
                z_ae = z_ae.view(z_ae.size(0), 512, 1, 1)  # [B, 512, 1, 1]
            # UNI特徴量抽出
            with torch.no_grad():
                features_uni = uni(x)  # [B, 1024]
            # UNI→AE特徴量
            features_ae = uni2ae(features_uni)  # [B, 512, 1, 1]
            # 損失計算（特徴量同士のMSE）
            loss = criterion(features_ae, z_ae)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch+1}] Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Mean Loss: {total_loss/(batch+1):.4f}")
        # チェックポイント保存
        torch.save(uni2ae.state_dict(), os.path.join(save_dir, f"uni2ae_epoch{epoch+1}.pth"))

if __name__ == '__main__':
    main()