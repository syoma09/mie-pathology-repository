#UNIencoderを用いて三重大学のデータセットを学習する。
#教師あり学習から自己教師あり学習に変更する。
#TCGAを利用したいが、入力画像が多すぎてどうすればいいかわからないので、一度三重大学で試す。
#出力は分類ではなく回帰で生存期間を予測するように変更。
#Loss関数はMSELossに変更。
import os
import datetime
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
from lifelines.utils import concordance_index

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download

# from survival import create_model
from aipatho.dataset import load_annotation
# from aipatho.metrics import MeanVarianceLoss
# from create_soft_labels import estimate_value, create_softlabel_tight, create_softlabel_survival_time_wise

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

def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2
    
    # dataset_root = Path(
    #     "/net/nfs3/export/home/sakakibara/data/TCGA_patch_image/" #TCGAのデータセット
    # )
    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=TumorMasking.FULL
    ) #三重大学のデータセット

    # road_root = Path("~/data/_out/mie-pathology/").expanduser()
    # log_root = Path("~/data/_out/mie-pathology/").expanduser() / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "uniencoder3")
    # annotation_path = Path("_data/survival_time_cls/TCGA_train_valid_44.csv").expanduser()
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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

    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is not on GPU")
        else:
            print(f"Parameter {name} is on GPU")

    print("モデルロード後:")
    print_gpu_memory_usage()
    dummy_input = torch.randn(32, 3, 256, 256).to(device)
    dummy_output = model(dummy_input)
    print(f"Model output shape: {dummy_output.shape}")

    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss()  # 回帰タスクのためMSELossを使用

    tensorboard = SummaryWriter(log_dir='./logs', filename_suffix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        model.train()
        train_loss = 0.
        train_mae = 0.
        train_index = 0.
        for batch, (x, y_true) in enumerate(train_loader):
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

            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
            print(f" {mae:.3}", end="")

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
            for batch, (x, y_true) in enumerate(valid_loader):
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