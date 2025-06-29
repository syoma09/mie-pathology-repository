#クラスタリングを行い自己教師あり学習を行うコード　ラベル付→再学習
import os
import datetime
from datetime import datetime
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
from lifelines.utils import concordance_index
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

from aipatho.dataset import load_annotation
from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import create_dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []

        for subject, label in annotations:
            self.__dataset += [
                (path, label)
                for path in (root / subject).iterdir()
            ]

        random.shuffle(self.__dataset)

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, item):
        path, label = self.__dataset[item]

        if os.path.isdir(path):
            return self.__getitem__((item + 1) % len(self.__dataset))
        try:
            img = Image.open(path).convert('RGB')
        except (OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            return self.__getitem__((item + 1) % len(self.__dataset))
        
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.float)

        return img, label

def print_gpu_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

def transform_image(img):
    if isinstance(img, torch.Tensor):
        return img.to(device)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return transform_image(img)

def extract_features(model, dataloader):
    model.eval()
    features = []
    indices = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc="Extracting features")):
            x = x.to(device)
            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)
            outputs = model.base_model(x)
            features.append(outputs.cpu().numpy())
            indices.extend(range(batch_idx * dataloader.batch_size, (batch_idx + 1) * dataloader.batch_size))
    return np.concatenate(features, axis=0), indices

def visualize_clusters(features, labels, save_path):
    reducer = umap.UMAP(n_components=2, random_state=0, n_jobs=-1) #n_jobs=-1で並列処理を有効にする
    reduced_features = reducer.fit_transform(features)

    # クラスタごとに異なる色を使用
    unique_labels = np.unique(labels)
    colors = list(mcolors.TABLEAU_COLORS.values())[:len(unique_labels)]

    plt.figure(figsize=(10, 10))
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], c=color, label=f'Cluster {label}', alpha=0.5)
    
    plt.legend()
    plt.title('UMAP visualization of clustered features')
    plt.savefig(save_path)
    plt.close()

def visualize_clusters_3d(features, labels, save_path): #3Dの可視化
    reducer = umap.UMAP(n_components=3, random_state=0, n_jobs=-1)
    reduced_features = reducer.fit_transform(features)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(reduced_features[mask, 0], reduced_features[mask, 1], reduced_features[mask, 2], 
                   color=colors(label), label=f'Cluster {label}', alpha=0.5)

    ax.legend()
    ax.set_title('3D UMAP visualization of clustered features')
    plt.savefig(save_path)
    plt.show()

def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2
    
    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    )

    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    #log_rootは以下に変更
    # log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_root = Path("~/data/_out/log_root/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S') #log_rootのパスを変更
    # log_root.mkdir(parents=True, exist_ok=True)
    
    # 実行IDとしてタイムスタンプを取得、実行ファイル名も取得
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") # 実行IDとしてタイムスタンプを取得
    script_name = Path(__file__).stem  # 実行時のファイル名を取得
    log_root = Path("~/data/_out/log_root/").expanduser() / f"{script_name}_{run_id}"
    log_root.mkdir(parents=True, exist_ok=True)

    annotation_path = Path("_data/survival_time_cls/20220726_cls.csv").expanduser()

    create_dataset(
        src=Path("/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None,
        target=TumorMasking.FULL
    )
    
    annotation = load_annotation(annotation_path)

    epochs = 1000
    batch_size = 32
    num_workers = os.cpu_count() // 4
    print(f"num_workers = {num_workers}")

    flag = 0
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'], flag), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'], flag), batch_size=batch_size,
        num_workers=num_workers, drop_last=True
    )

    # 環境変数からトークンを取得
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    login(token)

    local_dir = "../assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
    os.makedirs(local_dir, exist_ok=True)
    model_file = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    class CustomModel(nn.Module):
        def __init__(self, base_model):
            super(CustomModel, self).__init__()
            self.base_model = base_model
            self.fc = nn.Linear(1024, 512)
            self.additional_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512, bias=True),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1, bias=True),  # 回帰タスクのため出力を1に変更
            )
            for param in self.base_model.parameters():
                param.requires_grad = False

        def forward(self, x):
            features = self.base_model(x)
            features = self.fc(features)
            output = self.additional_layers(features)
            return output

    model = CustomModel(model)
    model = model.to(device)

    print("モデルロード後:")
    print_gpu_memory_usage()
    dummy_input = torch.randn(32, 3, 256, 256).to(device)
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001)

    criterion = nn.MSELoss()  # 回帰タスクのためMSELossを使用

    tensorboard = SummaryWriter(log_dir='./logs', filename_suffix=datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    scaler = torch.cuda.amp.GradScaler()

    # 特徴抽出フェーズ
    print("特徴抽出...")
    features, indices = extract_features(model, train_loader)
    print(f"Features shape: {features.shape}")

    #特徴量の次元削減 (UMAP) →なしでやってみる 1024次元のまま
    reducer = umap.UMAP(n_components=50, random_state=0)
    reduced_features = reducer.fit_transform(features)
    # reduced_features = features

    # クラスタリングフェーズ
    print("クラスタリング...")
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=10, random_state=0).fit(reduced_features) #n_clustersはクラスタ数　10から変更する
    pseudo_labels = kmeans.labels_
    silhouette_avg = silhouette_score(reduced_features, pseudo_labels)
    print(f"Kmeansシルエットスコア: {silhouette_avg}")

    # DBSCANクラスタリング
    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(reduced_features)
    dbscan_labels = dbscan.labels_
    dbscan_silhouette_avg = silhouette_score(reduced_features, dbscan_labels)
    print(f"DBSCANシルエットスコア: {dbscan_silhouette_avg}")

    #二次元可視化
    visualize_clusters(reduced_features, pseudo_labels, os.path.join(log_root, 'kmeans_cluster_visualization.png'))#data outの方各日にちにこの次元削減した２つの画像が入っている
    visualize_clusters(reduced_features, dbscan_labels, os.path.join(log_root, 'dbscan_cluster_visualization.png'))
    
    # Silhouetteスコアをファイルに保存
    output_file = os.path.join(log_root, 'silhouette_scores.txt')
    with open(output_file, 'w') as f:
        f.write(f"KMeans Silhouette Score: {silhouette_avg}\n")
        f.write(f"DBSCAN Silhouette Score: {dbscan_silhouette_avg}\n")
    print(f"Silhouette scores saved to: {output_file}")

    # 患者番号のリストを取得
    patients_df = pd.read_csv("_data/survival_time_cls/20220726_cls.csv")
    patient_numbers = patients_df['number'].astype(str).tolist()

    # 各患者のクラスタリング結果を保存
    for patient_number in patient_numbers:
        patchlist_path = f"/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/{patient_number}/patchlist/patchlist_updated.csv"
        if os.path.exists(patchlist_path):
            result_df = pd.read_csv(patchlist_path)
            result_df['kmeans_cluster'] = np.nan
            result_df['dbscan_cluster'] = np.nan
            for idx, (kmeans_label, dbscan_label) in zip(indices, zip(pseudo_labels, dbscan_labels)):
                if idx < len(result_df):
                    result_df.at[idx, 'kmeans_cluster'] = kmeans_label
                    result_df.at[idx, 'dbscan_cluster'] = dbscan_label
            result_df.to_csv(os.path.join(log_root, f'{patient_number}_clustered_patches.csv'), index=False)
            print(f"Clustered patches saved for patient {patient_number}")

            
     # 特定の患者番号を指定
    target_patient_number = patient_numbers[0]  # 最初の患者番号を使用（任意の患者番号に変更可能）

    # 特定の患者のデータを抽出
    patchlist_path = f"/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/{target_patient_number}/patchlist/patchlist_updated.csv"
    if os.path.exists(patchlist_path):
        result_df = pd.read_csv(patchlist_path)
        result_df['kmeans_cluster'] = 0
        result_df['dbscan_cluster'] = 0
        for idx, (kmeans_label, dbscan_label) in zip(indices, zip(pseudo_labels, dbscan_labels)):
            result_df.at[idx, 'kmeans_cluster'] = kmeans_label
            result_df.at[idx, 'dbscan_cluster'] = dbscan_label
        result_df.to_csv(os.path.join(log_root, f'{target_patient_number}_clustered_patches.csv'), index=False)
        print(f"Clustered patches saved for patient {target_patient_number}")

        # 特定の患者のデータを使用してUMAPによる可視化を行う
        patient_features = features[result_df.index]
        patient_labels = pseudo_labels[result_df.index]
        visualize_clusters_3d(patient_features, patient_labels, os.path.join(log_root, f'{target_patient_number}_kmeans_cluster_visualization_3d.png'))
        patient_labels = dbscan_labels[result_df.index]
        visualize_clusters_3d(patient_features, patient_labels, os.path.join(log_root, f'{target_patient_number}_dbscan_cluster_visualization_3d.png'))
    
    ###今やるべき課題はここまで　この下はそのあと ↑3dじゃなくて普通に可視化してみる　ミスりそう

    # 擬似ラベルを使用してデータセットを再構築
    class PseudoLabelDataset(torch.utils.data.Dataset):
        def __init__(self, root, annotations, pseudo_labels, flag):
            super(PseudoLabelDataset, self).__init__()
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            self.__dataset = []

            for idx, (subject, label) in enumerate(annotations):
                self.__dataset += [
                    (path, pseudo_labels[idx])
                    for path in (root / subject).iterdir()
                ]

            random.shuffle(self.__dataset)

        def __len__(self):
            return len(self.__dataset)

        def __getitem__(self, item):
            path, label = self.__dataset[item]

            if os.path.isdir(path):
                return self.__getitem__((item + 1) % len(self.__dataset))
            try:
                img = Image.open(path).convert('RGB')
            except (OSError, IOError) as e:
                print(f"Error loading image {path}: {e}")
                return self.__getitem__((item + 1) % len(self.__dataset))
            
            img = self.transform(img)
            label = torch.tensor(label, dtype=torch.float)

            return img, label

    train_loader = torch.utils.data.DataLoader(
        PseudoLabelDataset(dataset_root, annotation['train'], pseudo_labels, flag), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    # 微調整フェーズ
    for epoch in range(epochs):
        print(f"Finetuning Epoch [{epoch:5}/{epochs:5}]:")
        model.train()
        train_loss = 0.
        train_mae = 0.
        train_index = 0.
        for batch, (x, y_true) in enumerate(tqdm(train_loader, desc="Training", leave=False, dynamic_ncols=True)): #leave=Falseでプログレスバーを消す
            optimizer.zero_grad()
            x = x.to(device)
            y_true = y_true.to(device)

            x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
            x = torch.stack([transform_image(img) for img in x]).to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(x)
                y_pred = outputs.squeeze()

                loss = criterion(y_pred, y_true)

                if torch.isnan(loss):
                    print("Warning: NaN detected in total loss.")
                    continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = y_pred.cpu().data.numpy()
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            status = np.ones(len(y_true))
            index = concordance_index(y_true.cpu().numpy(), pred, status)

            train_loss += loss.item() / len(train_loader)
            train_mae += mae / len(train_loader)
            train_index += index / len(train_loader)
            
            print(f"\r  Batch({batch:6}/{len(train_loader):6})[{'=' * (30 * batch // len(train_loader)) + ' ' * 30}]: loss={loss.item():.4} {mae:.3}", end="")

            # print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
            #     batch, len(train_loader),
            #     ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
            #     loss.item()
            # ), end="")
            # print(f" {mae:.3}", end="")

        print("    train MV: {:3.3}".format(train_loss))
        print('')
        print("    train MAE: {:3.3}".format(train_mae))
        print('')
        print("    train INDEX: {:3.3}".format(train_index))
        print('')
        print('    Saving model...')
        torch.save(model.state_dict(), log_root / f"model{epoch:05}.pth")

        model.eval()
        metrics = {
            'train': {
                'loss': 0.,
            }, 'valid': {
                'loss': 0.,
            }
        }

        with torch.no_grad():
            valid_mae = 0.
            valid_index = 0.
            for batch, (x, y_true) in enumerate(tqdm(valid_loader, desc="Validation")):
                x, y_true = x.to(device), y_true.to(device)

                x = [transform(img) if isinstance(img, (Image.Image, np.ndarray)) else img for img in x]
                x = torch.stack([transform_image(img) for img in x]).to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    y_pred = outputs.squeeze()

                    loss = criterion(y_pred, y_true)

                if torch.isnan(loss):
                    print("Warning: NaN loss detected.")
                    continue
                
                pred = y_pred.cpu().data.numpy()
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                status = np.ones(len(y_true))
                index = concordance_index(y_true.cpu().numpy(), pred, status)

                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_mae += mae / len(valid_loader)
                valid_index += index / len(valid_loader)

                print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item()
                ), end="")

        print("    valid MV: {:3.3}".format(metrics['valid']['loss']))
        print('')
        print("    valid MAE: {:3.3}".format(valid_mae))
        print('')
        print("    valid INDEX: {:3.3}".format(valid_index))

        tensorboard.add_scalar('train_MV', train_loss, epoch)
        tensorboard.add_scalar('train_MAE', train_mae, epoch)
        tensorboard.add_scalar('train_Index', train_index, epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        tensorboard.add_scalar('valid_Index', valid_index, epoch)

if __name__ == '__main__':
    main()