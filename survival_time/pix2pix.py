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

from scipy.special import softmax
from collections import OrderedDict
from dataset_path import load_annotation, get_dataset_root_path
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 70*70patchGAN識別器モデルの定義
        # 2つの画像を結合したものが入力となるため、チャンネル数は3*2=6となる
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            self.__layer(64, 128),
            self.__layer(128, 256),
            self.__layer(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def __layer(self, input, output, stride=2):
        # DownSampling
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=4, stride=stride, padding=1),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # U-netのEncoder部分
        self.down0 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.down1 = self.__encoder_block(64, 128)
        self.down2 = self.__encoder_block(128, 256)
        self.down3 = self.__encoder_block(256, 512)
        self.down4 = self.__encoder_block(512, 512)
        self.down5 = self.__encoder_block(512, 512)
        self.down6 = self.__encoder_block(512, 512)
        self.down7 = self.__encoder_block(512, 512, use_norm=False)

        # U-netのDecoder部分
        self.up7 = self.__decoder_block(512, 512)
        self.up6 = self.__decoder_block(1024, 512, use_dropout=True)
        self.up5 = self.__decoder_block(1024, 512, use_dropout=True)
        self.up4 = self.__decoder_block(1024, 512, use_dropout=True)
        self.up3 = self.__decoder_block(1024, 256)
        self.up2 = self.__decoder_block(512, 128)
        self.up1 = self.__decoder_block(256, 64)
        # Gの最終出力
        self.up0 = nn.Sequential(
            self.__decoder_block(128, 3, use_norm=False),
            nn.Tanh(),
        )

    def __encoder_block(self, input, output, use_norm=True):
        # LeakyReLU+Downsampling
        layer = [
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1)
        ]
        # BatchNormalization
        if use_norm:
            layer.append(nn.BatchNorm2d(output))
        return nn.Sequential(*layer)

    def __decoder_block(self, input, output, use_norm=True, use_dropout=False):
        # ReLU+Upsampling
        layer = [
            nn.ReLU(True),
            nn.ConvTranspose2d(input, output, kernel_size=4,
                               stride=2, padding=1)
        ]
        # BatchNormalization
        if use_norm:
            layer.append(nn.BatchNorm2d(output))
        # Dropout
        if use_dropout:
            layer.append(nn.Dropout(0.5))
        return nn.Sequential(*layer)

    def forward(self, x):
        # 偽画像の生成
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        y7 = self.up7(x7)
        # Encoderの出力をDecoderの入力にSkipConnectionで接続
        y6 = self.up6(self.concat(x6, y7))
        y5 = self.up5(self.concat(x5, y6))
        y4 = self.up4(self.concat(x4, y5))
        y3 = self.up3(self.concat(x3, y4))
        y2 = self.up2(self.concat(x2, y3))
        y1 = self.up1(self.concat(x1, y2))
        y0 = self.up0(self.concat(x0, y1))

        return y0

    def concat(self, x, y):
        # 特徴マップの結合
        return torch.cat([x, y], dim=1)


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        # Real/Fake識別関数の損失を、シグモイド+バイナリクロスエントロピーで計算
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, is_real):
        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return self.loss(prediction, target_tensor.expand_as(prediction))


def create_dataset(
    src: Path, dst: Path,
    annotation: Path,
    size, stride,
    index: int = None, region: int = None
):
    # Lad annotation
    df = pd.read_csv(annotation)
    # print(df)

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
        delayed(save_patches)(path_svs, path_xml, base,
                              size, stride, resize, index, region)
        for path_svs, path_xml, base, size, stride, resize in args
    ])
    # print('args',args)


def update_learning_rate(self):
    # 学習率の更新、毎エポックごとに呼ばれる
    self.schedulerG.step()
    self.schedulerD.step()


def modify_learning_rate(self, epoch):
    # 学習率の計算
    if self.config.epochs_lr_decay_start < 0:
        return 1.0

    delta = max(0, epoch - self.config.epochs_lr_decay_start) / \
        float(self.config.epochs_lr_decay)
    return max(0.0, 1.0 - delta)


def weight_init(self, m):
    # パラメータ初期値の設定
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weght.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weght.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    patch_size = 256, 256
    stride = 512, 512
    index = 2
    # patch_size = 256, 256
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index=index
    )

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / \
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    batch_size = 32     # 64 requires 19 GiB VRAM
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

    # 生成器Gのオブジェクト取得とデバイス設定
    netG = Generator().to(device)
    # ネットワークの初期化
    # self.netG.apply(self.__weights_init)

    # 識別器Dのオブジェクト取得とデバイス設定
    netD = Discriminator().to(device)
    # Dのネットワークの初期化
    # self.netD.apply(self.__weights_init)

    # オプティマイザの初期化
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002)
    # 目的(損失関数)の設定
    # GAN損失(Adcersarial損失)
    criterionGAN = GANLoss().to(device)
    # L1損失
    criterionL1 = nn.L1Loss()

    """# 学習率スケジューラ設定
    schedulerG = torch.optim.lr_scheduler.LambdaLR(
        optimizerG, modify_learning_rate(epoch))
    schedulerD = torch.optim.lr_scheduler.LambdaLR(
        optimizerD, modify_learning_rate(epoch))"""

    #training_start_time = time.time()

    tensorboard = SummaryWriter(log_dir='logs')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )

    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        netD.train()
        netG.train()
        train_loss_D = 0.
        train_loss_G = 0.
        for batch, data in enumerate(train_loader):
            # ドメインAのラベル画像とドメインBの正解画像を設定
            realA = data[0].to(device)
            realB = data[1].to(device)

            # 生成器Gで画像生成
            fakeB = netG(realA)

            # 識別器Dの学習開始
            # 条件画像(A)と生成画像(B)を結合
            fakeAB = torch.cat((realA, fakeB), dim=1)
            pred_fake = netD(fakeAB.detach())
            # 偽画像を入力したときの識別機DのGAN損失を算出
            lossD_fake = criterionGAN(pred_fake, False)

            # 条件画像(A)と正解画像(B)を結合
            realAB = torch.cat((realA, realB), dim=1)
            # 識別機Dに正解画像を入力
            pred_real = netD(realAB)
            # 正解画像を入力したときの識別機DのGAN損失を算出
            lossD_real = criterionGAN(pred_real, True)

            # 偽画像と正解画像のGAN損失の合計に0.5を掛ける
            lossD = (lossD_fake + lossD_real) * 0.5

            # Dの勾配を0に設定
            optimizerD.zero_grad()
            # Dの逆伝播を計算
            lossD.backward()
            # Dの重みを更新
            optimizerD.step()

            train_loss_D += lossD.item() / len(train_loader)
            print("\r Batch({:6}/{:6})[{}]: loss={:4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                lossD.item()
            ), end="")
            # 生成器Gの学習開始
            # 識別器Dに生成画像を入力
            with torch.no_grad():
                pred_fake = netD(fakeAB)

            # 生成器GのGAN損失を算出
            lossG_GAN = criterionGAN(pred_fake, True)
            # 生成器GのL1損失を算出
            lossG_L1 = criterionL1(fakeB, realB)

            # 生成器Gの損失を合計
            lossG = lossG_GAN + lossG_L1

            # Gの勾配を0に設定
            optimizerG.zero_grad()
            # Gの逆伝播を計算
            lossG.backward()
            # Gの重みを更新
            optimizerG.step()
            train_loss_G += lossG.item() / len(train_loader)
            print("\r Batch({:6}/{:6})[{}]: loss={:4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                lossG.item()
            ), end="")
        print(train_loss_D)
        print('')
        print(train_loss_G)
        print('')
        print('   Saving model...')
        torch.save(netD.state_dict(), log_root / f"{model_name}{epoch:05}.pth")

if __name__ == "__main__":
    main()
