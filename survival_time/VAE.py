import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import math
import numpy
import random
import pandas as pd
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from joblib import Parallel, delayed
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from cnn.metrics import ConfusionMatrix
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from data.svs import save_patches


# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'
if torch.cuda.is_available():
    cudnn.benchmark = True


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations, flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((299, 299)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.__dataset = []
        self.paths = []
        for subject in annotations:
            self.paths += [
                path  # Same label for one subject
                for path in (root / subject).iterdir()
            ]
        if (flag == 0):
            self.__dataset += random.sample(self.paths, len(self.paths))
            #self.__dataset += random.sample(self.paths,1000)
        else:
            self.__dataset += random.sample(self.paths, flag)
        #self.__dataset += random.sample(self.paths,len(self.paths))

        # Random shuffle
        random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))

        '''self.paths = []
        for subject in subjects:
            print(subject)
            path = []
            path += list((root / subject).iterdir())
            if(subject == "57-10" or subject == "57-11"):
                self.paths += random.sample(path,4000)
            elif(subject == "38-4" or subject == "38-5"):
                self.paths += random.sample(path,len(path))
            elif(len(path) < 2000):
                self.paths += random.sample(path,len(path))
            else:
                self.paths+= random.sample(path,2000)
            self.paths += list((root / subject).iterdir())'''
        # print(self.paths[0])
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
        path = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)
        return img

    # @classmethod
    # def load_list(cls, root):
    #     # 顎骨正常データ取得と整形
    #
    #     with open(root, "rb") as f:
    #         output = pickle.load(f)
    #
    #     return output
    #
    # @classmethod
    # def load_torch(cls, _list):
    #     output = torch.cat([_dict["data"].view(1, 3, 32, 32) for _dict in _list],
    #                        dim=0)
    #
    #     return output
    #
    # @classmethod
    # def load_necrosis(cls, root):
    #     data = cls.load_list(root)
    #     data = cls.load_torch(data)
    #
    #     return data

# class WeightedProbLoss(nn.Module):
#     def __init__(self, classes):
#         super(WeightedProbLoss, self).__init__()
#
#         if isinstance(classes, int):
#             classes = [i for i in range(classes)]
#
#         self.classes = torch.Tensor(classes).to(device)
#
#     def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
#         """
#
#         :param pred:    Probabilities of each class
#         :param true:    1-hot vector
#         :return:
#         """
#
#         c_pred = torch.sum(torch.mul(pred, self.classes))
#         c_true = torch.argmax(true)


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=False)
        newnet = torch.nn.Sequential(*(list(net.children())[:-1]))
        self.enc = newnet
        self.enc_ave = nn.Linear(512, z_dim)  # average
        self.enc_dev = nn.Linear(512, z_dim)  # log(sigma^2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.enc(x)
        x_flat = torch.flatten(x, 1)
        ave = self.enc_ave(x_flat)  # average
        log_dev = self.enc_dev(x_flat)  # log(sigma^2)

        ep = torch.randn_like(ave)  # 平均0分散1の正規分布に従い生成されるz_dim次元の乱数
        z = ave + torch.exp(log_dev / 2) * ep  # 再パラメータ化トリック
        return z, ave, log_dev


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.reverse_flatten = nn.Linear(z_dim, 512*1*1)
        self.up7 = self.__decoder_block(512, 512)
        self.up6 = self.__decoder_block(512, 1024, use_norm=True)
        self.up5 = self.__decoder_block(1024, 512, use_norm=True)
        self.up4 = self.__decoder_block(512, 512, use_norm=True)
        self.up3 = self.__decoder_block(512, 256)
        self.up2 = self.__decoder_block(256, 128)
        self.up1 = self.__decoder_block(128, 64)

        # Gの最終出力
        self.up0 = nn.Sequential(
            self.__decoder_block(64, 3, use_norm=False),
            nn.Sigmoid()
            #nn.Tanh()
        )

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

    def forward(self, z):
        y9 = self.reverse_flatten(z)
        n, c = y9.shape
        y8 = y9.reshape(n, 512, 1, 1)
        y7 = self.up7(y8)
        y6 = self.up6(y7)
        y5 = self.up5(y6)
        y4 = self.up4(y5)
        y3 = self.up3(y4)
        y2 = self.up2(y3)
        y1 = self.up1(y2)
        y0 = self.up0(y1)
        y0 = torch.flatten(y0,1)
        return y0


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z, ave, log_dev = self.encoder(x)
        x = self.decoder(z)
        return x, z, ave, log_dev
        #return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ToImg(nn.Module):
    def forward(self, x):
        n, c = x.shape
        return x.reshape(n, 512, 1, 1)


def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        index: int = None, region: int = None
):
    print('index', index)
    # Lad annotation
    df = pd.read_csv(annotation)
    # print(df)
    args = []
    for _, subject in df.iterrows():
        number = subject['number']
        subject_dir = dst / str(number)
        if not subject_dir.exists():
            subject_dir.mkdir(parents=True, exist_ok=True)
        '''else:
            print(f"Subject #{number} already exists. Skip.")
            continue'''

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
    #mem_total = 80
    #n_jobs = int(mem_total / 20)
    n_jobs = 3
    print(f'Process in {n_jobs} threads.')
    # Parallel execution
    Parallel(n_jobs=n_jobs)([
        delayed(save_patches)(path_svs, path_xml, base,
                              dst, size, stride, resize, index, region)
        for path_svs, path_xml, base, size, stride, resize in args
    ])
    # print('args',args)


def criterion(predict, target, ave, log_dev):
#def criterion(predict, target, ave, log_dev):
    mse_loss = nn.functional.mse_loss(predict, target, reduction='sum')
    r_loss = mse_loss
    #kl_loss = -1.0 * weight * torch.sum(1 + log_dev - ave.pow(2) - log_dev.exp())
    kl_loss = -0.5 * torch.sum(1 + log_dev - ave.pow(2) - log_dev.exp())
    loss = r_loss + kl_loss
    #loss = r_loss
    #print("kl_loss:",kl_loss)
    return r_loss,kl_loss,loss
    #return loss


def main():
    #patch_size = 512,512
    patch_size = 256, 256
    stride = 256, 256
    #stride = 128,128
    index = 2
    # patch_size = 256, 256
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index=index
    )

    dataset_root_not = get_dataset_root_not_path(
        patch_size=patch_size,
        stride=stride,
        index=index
    )

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / \
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        "../_data/survival_time_cls/20221206_Auto.csv"
        # "../_data/survival_time_cls/20220725_aut1.csv"
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
    )

    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root_not,
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
    train_dataset = []
    valid_dataset = []
    flag = 0
    train_dataset.append(PatchDataset(dataset_root, annotation['train'], flag))
    flag = len(train_dataset[0])
    train_dataset.append(PatchDataset(
        dataset_root_not, annotation['train'], flag))
    flag = 0
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid'], flag))
    flag = len(valid_dataset[0])
    valid_dataset.append(PatchDataset(
        dataset_root_not, annotation['valid'], flag))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    print("train_loader:", len(train_loader))
    print("valid_loader:", len(valid_loader))
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    z_dim = 512
    net = VAE(z_dim).to(device)
    # print(net)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.RAdam(net.parameters(),lr = 0.001)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()

    tensorboard = SummaryWriter(log_dir='./logs_VAE')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    criterion2 = nn.MSELoss()
    numepoch = 0
    for epoch in range(epochs):
        numepoch += 1
        print(f"Epoch [{epoch:5}/{epochs:5}]:")

        # Initialize metric values on epoch
        metrics = {
            'train': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }, 'valid': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }
        }

        # Switch to training mode
        net.train()

        for batch, (x) in enumerate(train_loader):
            #input = x.to(device).view(-1, 3*256*256).to(torch.float32)
            input = x.to(device)
            output, z, ave, log_dev = net(input)   # Forward
            #print(output.shape)
            input_true = x.to(device).view(-1, 3*256*256).to(torch.float32)
            #print(input_true[0])
            #print(output[0])
            #print(input_true.shape)
            # Calculate training loss
            #r_loss,kl_loss,loss = criterion(output, input, ave,log_dev)
            #loss = criterion(output, input, ave, log_dev)
            loss = criterion(output, input_true, ave, log_dev)
            #loss = criterion2(input,output)
            optimizer.zero_grad()
            loss.backward()     # Backward propagation
            optimizer.step()    # Update parameters

            # Logging
            metrics['train']['loss'] += loss.item() / len(train_loader)
            # metrics['train']['cmat'] += ConfusionMatrix(
            #    y_pred.cpu(), y_true.cpu())
            # Screen output
            #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} r_loss={:4} kl_loss={:4}".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4}".format(    
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                #loss.item(),r_loss,kl_loss
                loss.item()
            ), end="")

        print('')
        print('  Saving model...')
        torch.save(net.state_dict(), log_root / f"model{epoch:05}.pth")

        # Switch to evaluation mode
        net.eval()

        # Calculate validation metrics
        with torch.no_grad():
            for i, (x) in enumerate(valid_loader):
                input = x.to(device)
                output, z, ave, log_dev = net(input)   # Forward
                #print("inp:",input)
                #print("out:",output)
                input_true = x.to(device).view(-1, 3*256*256).to(torch.float32)
                # Calculate training loss
                #r_loss,kl_loss,loss = criterion(output, input, ave, log_dev,epoch)
                #loss = criterion(output, input, ave, log_dev)
                loss = criterion(output, input_true, ave, log_dev)
                #loss = criterion2(input,output)
                # print(loss.item())
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                #metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                
                #print("\r  Validating... ({:6}/{:6})[{}]: loss={:.4} r_loss={:4} kl_loss={:4}".format(
                print("\r  Validating... ({:6}/{:6})[{}]: loss={:.4}".format(
                    i, len(valid_loader),
                    ('=' * (30 * i // len(valid_loader)) + " " * 30)[:30],
                    #loss.item(),r_loss,kl_loss
                    loss.item()
                ), end="")

        # Console write
        print("")
        print("    train loss  : {:3.3}".format(metrics['train']['loss']))
        # print("          f1inv : {:3.3}".format(
        #    metrics['train']['cmat'].f1inv))
        #print("          npv   : {:3.3}".format(metrics['train']['cmat'].npv))
        #print("          tnr   : {:3.3}".format(metrics['train']['cmat'].tnr))
        # print(metrics['train']['cmat'])
        print("    valid loss  : {:3.3}".format(metrics['valid']['loss']))
        # print("          f1inv : {:3.3} (f1={:3.3})".format(
        #    metrics['valid']['cmat'].f1inv,
        #    metrics['valid']['cmat'].f1
        # ))
        #print("          npv   : {:3.3}".format(metrics['valid']['cmat'].npv))
        #print("          tnr   : {:3.3}".format(metrics['valid']['cmat'].tnr))
        #print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard
        for tv in ['train', 'valid']:
            # Loss
            tensorboard.add_scalar(f"{tv}_loss", metrics[tv]['loss'], numepoch)
            # For ConfusionMatrix
            """for m_name in ['f1', "f1inv", "npv", "tpr", "precision", "recall", "tn", "tp", "fn", "fp"]:
                tensorboard.add_scalar(f"{tv}_{m_name}", getattr(
                    metrics[tv]['cmat'], m_name), epoch)"""


if __name__ == '__main__':
    main()
