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
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from joblib import Parallel, delayed
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from function import load_annotation, get_dataset_root_path
from data.svs import save_patches


# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.__dataset = []
        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
        ]
        # Random shuffle
        random.shuffle(self.__dataset)
        self.__num_class = 2
        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([l for _, l in self.__dataset if l <= 11]))
        print('  # of 1  :', len([l for _, l in self.__dataset if (11 < l) & (l <= 22)]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))
        
    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def load_datasets():
    tumor_transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,))
    ])
    
    non_tumor_transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5,), std=(0.5,0.5,0.5,))
    ])
    tumor_trainsets = datasets.ImageFolder(root = '',transform=tumor_transform)
    non_tumor_trainsets = datasets.ImageFolder(root = '',transform=non_tumor_transform)
    Image_datasets = ConcatDataset(tumor_trainsets,non_tumor_trainsets)
    
    return 
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



def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        index: int = None, region: int = None
):
    print('index',index)
    # Lad annotation
    df = pd.read_csv(annotation)
    #print(df)
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
    patch_size = 1024,1024
    stride = 512,512
    index = 1
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
        "../_data/survival_time_cls/20221206_Auto.csv"
    ).expanduser()
    # Create dataset if not exists
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )
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
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    net = resnet18(pretrained = False)
    net.fc = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        #nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        #nn.ReLU(),
        nn.Linear(512, 2, bias=True),
        nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    print(net)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()

    criterion = nn.MSELoss()
    tensorboard = SummaryWriter(log_dir='./logss')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    print(net)
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        for batch, (x) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y_pred = net(x)   # Forward
            #print(y_pred)
            #print(y_pred)
            #print("yp:", y_pred)
            loss = criterion(y_pred,x)
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            # Logging
            train_loss += loss.item() / len(train_loader)
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
        print("train_loss",train_loss)
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), log_root / f"{model_name}{epoch:05}.pth")
        # Switch to evaluation mode
        net.eval()
        # On training data
        # Initialize validation metric values
        metrics = {
            'train': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }, 'valid': {
                'loss': 0.,
                'cmat': ConfusionMatrix(None, None)
            }
        }
        # Calculate validation metrics
        with torch.no_grad():
            valid_loss=0.
            for batch, (x) in enumerate(valid_loader):
                x = x.to(device)
                y_pred = net(x)  # Prediction
                loss = criterion(y_pred,x)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
            """for x, y_true in train_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = net(x)  # Prediction
                metrics['train']['loss'] += criterion(y_pred, y_true).item() / len(train_loader)
                # metrics['train']['cmat'] += ConfusionMatrix(y_pred, y_true)"""
        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        #with open("valid_loss.txt") as f:
        #print("epoch: {:3.3}    valid loss: {:3.3}".format(epoch,metrics['valid']['loss']), file = f)
        print("    valid loss: {:3.3}".format(metrics['valid']['loss']))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard'''
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
