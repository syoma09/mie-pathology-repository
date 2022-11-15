import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import statistics
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile

from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from collections import OrderedDict
from function import load_annotation, get_dataset_root_path
from data.svs import save_patches

# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True
class ColorAndGray(object):
    def __call__(self, img):
        # ToTensor()の前に呼ぶ場合はimgはPILのインスタンス
        gray = img.convert("L")
        return img, gray

class MultiInputWrapper(object):
    def __init__(self, base_func):
        self.base_func = base_func

    def __call__(self, xs):
        if isinstance(self.base_func, list):
            return [f(x) for f, x in zip(self.base_func, xs)]
        else:
            return [self.base_func(x) for x in xs]

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.__dataset = []
        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
        ]
        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print(len(self.__dataset))
    def __len__(self):
        return len(self.__dataset)
    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
    # img = self.data[item, :, :, :].view(3, 32, 32)
        path, label = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        gray = img.convert("L")
        img = self.transform(img)
        gray = self.transform(gray)
        return img, gray

def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        index: int = None, region: int = None
):
    # Lad annotation
    df = pd.read_csv(annotation)
    #print(df)

    args = []
    for _, subject in df.iterrows():
        number = subject['number']
        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Subject #{number} already exists. Skip.")
            continue
        
        path_svs = src / f"{number}.svs"
        path_xml = src / f"{number}.xml"
        if not path_svs.exists() or not path_xml.exists():
            print(f"{path_svs} or {path_xml} do not exists.")
            continue

        base = subject_dir / 'patch'
        resize = 256, 256
        args.append((path_svs, path_xml, base, size, stride, resize))
        # # Serial execution
        # save_patches(path_svs, path_xml, base, size=size, stride=stride)

    # Approx., 1 thread use 20GB
    # n_jobs = int(mem_total / 20)
    n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base, size, stride, resize, index, region)
        for path_svs, path_xml, base, size, stride, resize in args
    ])
    #print('args',args)

def main():
    patch_size = 256, 256
    stride = 512, 512
    index = 2
    # patch_size = 256, 256
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )
    
    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        "../_data/survival_time_cls/20220725_aut1.csv"
    ).expanduser()
    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    # Existing subjects are ignored in the function
    """create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )"""
    # Load annotations
    annotation = load_annotation(annotation_path)
    # echo $HOME == ~
    #src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    # Write dataset on SSD (/mnt/cache/)
    #dataset_root = Path("/mnt/cache").expanduser()/ os.environ.get('USER') / 'mie-pathology'
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    epochs = 10000
    batch_size = 8     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    '''# Load train/valid yaml
    with open(src / "survival_time.yml", "r") as f:
        yml = yaml.safe_load(f)'''

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    #model
    net_G = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=1, out_channels=3, init_features=256, pretrained=False)
    net_D = torchvision.models.vgg16(pretrained=False)
    
    net_G, net_D = net_G.to(device),net_D.to(device)
    #optim
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=0.001)
    # エラー推移
    result = {}
    result["log_loss_G_sum"] = []
    result["log_loss_G_bce"] = []
    result["log_loss_G_mae"] = []
    result["log_loss_D"] = []

    
    # loss function
    bce_loss = nn.BCEWithLogitsLoss()
    mae_loss = nn.L1Loss()
    
    for i in range(epochs):
        # ロスを計算するためのラベル変数 (PatchGAN)
        ones = torch.ones(8, 3, 256, 256).to(device)
        zeros = torch.zeros(8, 1, 256, 256).to(device)
        log_loss_G_sum, log_loss_G_bce, log_loss_G_mae, log_loss_D = [], [], [], []
        for batch, (real_color, input_gray) in enumerate(train_loader):
            batch_len = len(real_color)
            real_color, input_gray = real_color.to(device), input_gray.to(device)

            # Gの訓練
            # 偽のカラー画像を作成
            fake_color = net_G(input_gray)

            # 偽画像を一時保存
            fake_color_tensor = fake_color.detach()

            # 偽画像を本物と騙せるようにロスを計算
            LAMBD = 100.0 # BCEとMAEの係数
            out = net_D(torch.cat([fake_color, input_gray], dim=1))
            loss_G_bce = bce_loss(out, ones[:batch_len])
            loss_G_mae = LAMBD * mae_loss(fake_color, real_color)
            loss_G_sum = loss_G_bce + loss_G_mae

            log_loss_G_bce.append(loss_G_bce.item())
            log_loss_G_mae.append(loss_G_mae.item())
            log_loss_G_sum.append(loss_G_sum.item())

            # 微分計算・重み更新
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            loss_G_sum.backward()
            optimizer_G.step()

            # Discriminatoの訓練
            # 本物のカラー画像を本物と識別できるようにロスを計算
            real_out = net_D(torch.cat([real_color, input_gray], dim=1))
            loss_D_real = bce_loss(real_out, ones[:batch_len])

            # 偽の画像の偽と識別できるようにロスを計算
            fake_out = net_D(torch.cat([fake_color_tensor, input_gray], dim=1))
            loss_D_fake = bce_loss(fake_out, zeros[:batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            # 微分計算・重み更新
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        result["log_loss_G_sum"].append(statistics.mean(log_loss_G_sum))
        result["log_loss_G_bce"].append(statistics.mean(log_loss_G_bce))
        result["log_loss_G_mae"].append(statistics.mean(log_loss_G_mae))
        result["log_loss_D"].append(statistics.mean(log_loss_D))
        print(f"log_loss_G_sum = {result['log_loss_G_sum'][-1]} " +
              f"({result['log_loss_G_bce'][-1]}, {result['log_loss_G_mae'][-1]}) " +
              f"log_loss_D = {result['log_loss_D'][-1]}")

        # 画像を保存
        if not os.path.exists("stl_color"):
            os.mkdir("stl_color")
        # 生成画像を保存
        torchvision.utils.save_image(fake_color_tensor[:min(batch_len, 100)],
                                f"stl_color/fake_epoch_{i:03}.png",
                                range=(-1.0,1.0), normalize=True)
        torchvision.utils.save_image(real_color[:min(batch_len, 100)],
                                f"stl_color/real_epoch_{i:03}.png",
                                range=(-1.0, 1.0), normalize=True)

        # モデルの保存
        if not os.path.exists("stl_color/models"):
            os.mkdir("stl_color/models")
        if i % 10 == 0 or i == 199:
            torch.save(model_G.state_dict(), f"stl_color/models/gen_{i:03}.pytorch")                        
            torch.save(model_D.state_dict(), f"stl_color/models/dis_{i:03}.pytorch")                        
        

if __name__ == "__main__":
    main()