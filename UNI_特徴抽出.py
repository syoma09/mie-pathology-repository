#20250617
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import os

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def extract_features_for_slide(slide_dir, base_model, transform, save_path):
    features = []
    patch_names = []
    patch_list = sorted(list(slide_dir.glob("*.png")))
    for i, patch_path in enumerate(tqdm(patch_list, desc=f"Extracting {slide_dir.name}", leave=False)):
        img = Image.open(patch_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = base_model(img)
        features.append(feat.squeeze(0).cpu())
        patch_names.append(patch_path.name)
    if features:
        features = torch.stack(features)  # (N_patches, feature_dim)
        torch.save({'features': features, 'patch_names': patch_names}, save_path)

def main():
    # モデル準備（UNIの部分のみ）
    import timm
    from huggingface_hub import login, hf_hub_download

    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    login(token)

    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)
    # 既にダウンロード済みならforce_download=FalseでOK
    model_file = hf_hub_download(
        "MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=local_dir,
        force_download=False
    )
    base_model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    base_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    base_model.eval()
    base_model.to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # パッチ画像ディレクトリ
    patch_root = get_cache_dir(patch=(256,256), stride=(256,256), target=TumorMasking.FULL)
    save_root = Path("/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/UNI_tokutyou")
    save_root.mkdir(parents=True, exist_ok=True)

    slide_dirs = [d for d in patch_root.iterdir() if d.is_dir()]
    for idx, slide_dir in enumerate(tqdm(slide_dirs, desc="Slides")):
        save_path = save_root / f"{slide_dir.name}.pt"
        if save_path.exists():
            tqdm.write(f"[{idx+1}/{len(slide_dirs)}] {slide_dir.name}: already extracted, skipping.")
            continue
        tqdm.write(f"[{idx+1}/{len(slide_dirs)}] {slide_dir.name}: extracting features...")
        extract_features_for_slide(slide_dir, base_model, transform, save_path)
        tqdm.write(f"[{idx+1}/{len(slide_dirs)}] {slide_dir.name}: done.")

if __name__ == "__main__":
    main()