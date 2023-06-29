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
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from data.svs import save_patches


# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        self.transform_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(256, 256)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])
        
        self.__dataset = []
        for subject in annotations:
            self.__dataset += [
                path   # Same label for one subject
                for path in (root / subject).iterdir()
        ]
            
        # Random shuffle
        random.shuffle(self.__dataset)
        
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
        path  = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img_aug = self.transform_aug(img)
        img = self.transform(img)
        if("not" in str(path)):
            true_class = 1
        else:
            true_class = 0
        # Tensor
        true_class = torch.tensor(true_class, dtype=torch.float)
        return img, img_aug, true_class
        
    
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
    def __init__(self):
        super(Encoder, self).__init__()
        net = torchvision.models.resnet18(pretrained=False)
        self.enc = torch.nn.Sequential(*(list(net.children())[:-1])) 
        
        
    def forward(self, x):
        x = self.enc(x)
        x = torch.flatten(x, 1)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        
        self.encoder = Encoder()
        
    def forward(self, x):
        x = self.encoder(x)
        return x

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels, augmented_embeddings):
        batch_size = embeddings.size(0)
        total_batch_size = batch_size * 2

        # 特徴ベクトル間の類似度行列を計算
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        augmented_sim_matrix = torch.matmul(augmented_embeddings, augmented_embeddings.t())

        # 正解クラスの対数確率を取得
        mask = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        positives = torch.cat([sim_matrix[mask], augmented_sim_matrix[mask]])

        # 正解クラス以外の類似度の合計を計算
        mask_neg = ~mask
        negatives = torch.cat([sim_matrix[mask_neg], augmented_sim_matrix[mask_neg]])

        # 全てのクラスの類似度をsoftmaxで正規化
        sim_matrix = torch.cat([sim_matrix, augmented_sim_matrix], dim=1) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # 最大値を引いて数値安定化
        sim_matrix = torch.exp(sim_matrix)

        # 損失を計算して平均を取る
        negatives_adjusted = negatives.unsqueeze(1).expand(-1, positives.shape[0])
        #negatives_adjusted = negatives.expand(-1, positives.shape[0])  # tensor2の列をtensor1の列数に拡張
        loss = -torch.log(positives / (torch.cat(positives,negatives_adjusted))).mean()
        print(loss.shape)

        return loss


def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

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
        delayed(save_patches)(path_svs, path_xml, base, dst, size, stride, resize, index, region)
        for path_svs, path_xml, base, size, stride, resize in args
    ])
    #print('args',args)

def main():
    patch_size = 256,256
    stride = 256,256
    index = 2
    # patch_size = 256, 256
    dataset_root = get_dataset_root_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )
    
    dataset_root_not = get_dataset_root_not_path(
        patch_size=patch_size,
        stride=stride,
        index = index
    )

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        "../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
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
    """train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )"""
    train_dataset = []
    valid_dataset = []
    train_dataset.append(PatchDataset(dataset_root, annotation['train']))
    train_dataset.append(PatchDataset(
        dataset_root_not, annotation['train']))
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid']))
    valid_dataset.append(PatchDataset(
        dataset_root_not, annotation['valid']))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    
    net = ContrastiveModel().to(device)
    print(net)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()

    criterion = SupervisedContrastiveLoss(temperature = 1.0)
    tensorboard = SummaryWriter(log_dir='./logs_Cont')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        for batch, (img,img_aug,labels) in enumerate(train_loader):
            optimizer.zero_grad()
            img, img_aug, labels = img.to(device), img_aug.to(device), labels.to(device)
            # 特徴抽出モデルに入力して特徴ベクトルを取得
            original_embedding = net(img)
            augmented_embedding = net(img_aug)
            loss = criterion(original_embedding, labels, augmented_embedding)
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
            for batch, (img,img_aug,labels) in enumerate(valid_loader):
                img, img_aug, labels = img.to(device), img_aug.to(device), labels.to(device)
                # 特徴抽出モデルに入力して特徴ベクトルを取得
                original_embedding = net(img)
                augmented_embedding = net(img_aug)
                loss = criterion(original_embedding, labels, augmented_embedding)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                print("\r  Validating... ({:6}/{:6})[{}]: loss={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    #loss.item(),loss1,loss2,loss3
                    loss.item()
                ), end="")
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