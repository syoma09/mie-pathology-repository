import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.utils.data
import torchvision
#import yaml
import timm
import math
import random
import numpy as np
import pandas as pd
import sys
import itertools
import matplotlib.pyplot as plt
from AutoEncoder import create_model
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from sklearn.cluster import KMeans
from collections import OrderedDict
from lifelines.utils import concordance_index
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from data.svs import save_patches
from AgeEstimation.mean_variance_loss import MeanVarianceLoss
from VAE import VAE
from Unet import Generator
from contrastive_learning import Hparams,SimCLR_pl,AddProjection
from labels import estimate_value, c_soft, hard_to_soft_labels
from sklearn.manifold import TSNE

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
            #torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize(
            #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #)
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.__dataset = []

        for subject, label in annotations:
            paths = []
            paths += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
            ]
            #print(paths)
            random.shuffle(paths)
            #print(paths)
            self.__dataset.append(paths) 
        #print(self.__dataset)
        self.__dataset = list(itertools.chain.from_iterable(self.__dataset))
        #print(self.__dataset)
        """
        if(flag == 1):
            paths = []
            for subject, label in annotations:
                paths += [
                    (path, label)   # Same label for one subject
                    for path in (root / subject).iterdir()
                ] 
            min = len(paths)
            for i in range(3):
                if(min > len([l for _, l in paths if (i * 11 < l) & (l <= (i+1) * 11)])):
                    min = len([l for _, l in paths if (i * 11 < l) & (l <= (i+1) * 11)])
            
            print(min)
            class_dataset_0 = [item for item in paths if item[1] <= 11]
            class_dataset_1 = [item for item in paths if 11 < item[1] and item[1] <= 22]
            class_dataset_2 = [item for item in paths if 22 < item[1] and item[1] <= 33]
            class_dataset_3 = [item for item in paths if 33 < item[1] and item[1] <= 44]
            
            self.__dataset += random.sample(class_dataset_0,min)
            self.__dataset += random.sample(class_dataset_1,min)    
            self.__dataset += random.sample(class_dataset_2,min)
            self.__dataset += random.sample(class_dataset_3,min)
        else:
            for subject, label in annotations:
                self.__dataset += [
                    (path, label)   # Same label for one subject
                    for path in (root / subject).iterdir()
            ]"""
            
        # Random shuffle
        #random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))
        self.__num_class = 4
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([l for _, l in self.__dataset if l <= 11]))
        print('  # of 1  :', len([l for _, l in self.__dataset if (11 < l) & (l <= 22)]))
        print('  # of 2  :', len([l for _, l in self.__dataset if (22 < l) & (l <= 33)]))
        print('  # of 3  :', len([l for _, l in self.__dataset if (33 < l) & (l <= 44)]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))




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
        #print(self.paths[0])
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
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        #label /= 90.
        '''if(label < 13):
            label_class = 0
        elif(label < 34):
            label_class = 1
        elif(label < 67):
            label_class = 2'''
        if(label < 11):
            label_class = 0
        elif(label < 22):
            label_class = 1
        elif(label < 33):
            label_class = 2
        elif(label < 44):
            label_class = 3
        '''elif(label < 44):
            label_class = 4
        elif(label < 36):
            label_class = 5
        elif(label < 42):
            label_class = 6
        elif(label < 48):
            label_class = 7
        elif(label < 24):
            label_class = 11
        elif(label < 26):
            label_class = 12
        elif(label < 28):
            label_class = 13
        elif(label < 30):
            label_class = 14
        elif(label < 32):
            label_class = 15
        elif(label < 34):
            label_class = 16
        elif(label < 36):
            label_class = 17
        elif(label < 38):
            label_class = 18
        elif(label < 40):
            label_class = 19
        elif(label < 42):
            label_class = 20
        elif(label < 44):
            label_class = 21
        elif(label < 46):
            label_class = 22
        elif(label < 48):
            label_class = 23
        elif(label < 50):
            label_class = 24
        elif(label < 52):
            label_class = 25
        elif(label < 54):
            label_class = 26
        elif(label < 56):
            label_class = 27
        elif(label < 58):
            label_class = 28
        elif(label < 60):
            label_class = 29
        elif(label < 62):
            label_class = 30
        elif(label < 64):
            label_class = 31
        elif(label < 66):
            label_class = 32
        elif(label < 68):
            label_class = 33'''
        # Tensor
        label = torch.tensor(label, dtype=torch.float)
        soft_labels = c_soft(label,4)
        return img, soft_labels, label, label_class

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
#
#         return torch.abs(c_pred - c_true)

"""def Mean_Variance_loss(y_pred,y_class):
    N = 4 #Number of classes
    #y_true one-hot-encording
    yt_one = np.zeros((len(y_class),N))
    for i in range (len(y_class)):
        yt_one[i][int(y_class[i])-1] = 1
                
    y_m = torch.zeros(len(y_class),device = device,dtype = torch.float32) 
    y_v = torch.zeros(len(y_class),device = device,dtype = torch.float32)
    y_pred_soft = torch.nn.functional.softmax(y_pred,dim=1)
    for i in range (len(y_pred_soft)):
        for j in range (1,N+1): # class label
            y_m[i] += j * y_pred_soft[i][j-1] # Eq.2
        for j in range (1,N+1): # class label
            y_v[i] += y_pred_soft[i][j-1] * pow(j-y_m[i],2) # Eq.3
                
    loss_mean = torch.zeros(len(y_class),device = device,dtype = torch.float32 ,requires_grad=False)
    for i in range (len(y_class)):
        for j in range(1,N+1):
            if(yt_one[i][j-1] == 1):
                loss_mean[i] = pow(y_m[i]-j,2) # Calculate mean loss Eq.4
    loss_s_tensor = torch.zeros(len(y_class),device = device,dtype = torch.float32 ,requires_grad=False)
                
    #Caliculate Softmax Loss Eq.6
    for i in range (len(y_class)):
        for j in range(N):
            if(yt_one[i][j] == 1):
                if(-1 * torch.log(y_pred[i][j]).item() == math.inf):
                    print(1)
                    loss_s_tensor[i] = torch.log(y_pred[i][j]+1.0E-8)
                else:
                    loss_s_tensor[i] = torch.log(y_pred[i][j])
                loss_s_tensor[i] = y_pred[i][j]
                
    #Type transformation numpy to tensor
    loss_m = torch.mean(loss_mean)/2.0
    loss_v = torch.mean(y_v)  # Calculate variance loss Eq.5
    loss_s = torch.mean(torch.nn.functional.log_softmax(loss_s_tensor,dim=0))
    lamda1 = 0.2
    lamda2 = 0.05
    loss = -1.0 * loss_s + lamda1 * loss_m + lamda2 * loss_v
            
    return loss"""

# Extracter + TransformerEncoder
class ExtTrans(torch.nn.Module):
    def __init__(self,ext ,est ,PE):
        super().__init__()
        self.ext = ext
        self.est = est
        self.PE = PE
    def forward(self, x):
        x = self.ext(x)
        sort_features,sort_clusters = features_sort(x)
        x = self.PE(sort_features,sort_clusters).to(device)
        x = self.est(x)
        #print(x.shape)
        return x

""" # 3 dimention PE
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        #self.register_buffer('pe', pe)

    def forward(self, x,cluster):
        
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        
        position = cluster.unsqueeze(1)
        pe = torch.zeros(len(cluster), 1, self.d_model).to(device)
        pe[:, 0, 0::2] = torch.sin(position * self.div_term)
        pe[:, 0, 1::2] = torch.cos(position * self.div_term)
        x = x*math.sqrt(self.d_model) + pe[:x.size(0)]
        return self.dropout(x)
        """
# 2 dimention PE
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000.0) / d_model))
        

    def forward(self, x,cluster):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
        """
        #claster = torch.tensor(claster)
        
        position = cluster.unsqueeze(1)
        pe = torch.zeros_like(x)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
        x = x*math.sqrt(self.d_model) + pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, 
                    nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        #self.pos_encoder = PositionalEncoding(claster, d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        

        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, src):
        """
        Args:
            src: Transformerへの入力データ
        Returns:
            Transformerの出力
        """
        
        #src = self.pos_encoder(src,claster)
        output = self.transformer_encoder(src)
        
        return output

def features_sort(features):
    clusters = KMeans(n_clusters = 4,n_init='auto').fit(features.cpu().detach().numpy())
    sort_clusters, sort_clusters_index = torch.sort(torch.tensor(clusters.labels_),dim=0)
    sort_clusters_index = sort_clusters_index.to(device)
    sort_features = features[sort_clusters_index]

    return sort_features, sort_clusters

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
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        "../_data/survival_time_cls/20220726_cls.csv"
        #"../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        #"../_data/survival_time_cls/fake_data.csv"
    ).expanduser()
    # Create dataset if not exists
    """if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)"""
    # Existing subjects are ignored in the function
    """create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )"""
    
    """create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root_not,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )"""
    # Load annotations
    annotation = load_annotation(annotation_path)
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    epochs = 1000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 4   # For SMT
    # Load train/valid yaml
    '''with open(src / "survival_time.yml", "r") as f:
        yml = yaml.safe_load(f)'''

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    
    flag = 0
    
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'],flag), batch_size=batch_size,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'],flag), batch_size=batch_size,
        num_workers=num_workers
    )
    """
    train_dataset = []
    valid_dataset = []
    flag = 0
    train_dataset.append(PatchDataset(dataset_root, annotation['train'],flag))
    flag = 1
    train_dataset.append(PatchDataset(dataset_root_not, annotation['train'],flag))
    flag = 0
    valid_dataset.append(PatchDataset(dataset_root, annotation['valid'],flag))
    flag =1
    valid_dataset.append(PatchDataset(dataset_root_not, annotation['valid'],flag))
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(valid_dataset), batch_size=batch_size,
        num_workers=num_workers
    )
    """
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    z_dim = 512
    #net = UNet_2D().to(device)
    #net = create_model()
    
    """
    # contrastive
    train_config = Hparams()
    ext = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512)
    
    ext.model.projection = nn.Sequential(
    #net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
    )
    
    ext.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        #road_root / "20230530_205222" /'20230530_205224model00020.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #road_root / "20230712_131831" /'20230712_131837model00140.pth', map_location=device)
        #road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
        road_root / "20230919_175330" /'20230919_175350model00125.pth', map_location=device) #AE
        #road_root / "20230725_173524" /'20230725_190953model00092.pth', map_location=device)
        #road_root / "20230907_132454" /'model00014.pth', map_location=device) #Cox                                 
    )"""
    """
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    #print(net.model.projection)
    
    #net.model.projection = nn.Sequential(
    net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )"""
    
    """net = nn.Sequential(
        net,
        nn.Sequential(
        nn.Linear(512, 1024, bias=True),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 2048, bias=True),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 1024, bias=True),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )
        #nn.Softmax(dim=1)
        # nn.Sigmoid()
    )"""
    """
    #for param in net.model.parameters():
    for param in net.parameters():
        param.requires_grad = False
    #print(net.model.backbone.conv1.bias)
    #last_layer = net.model.projection
    last_layer = net.dec
    #last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True
    """
    # ResNet
    """
    net = torchvision.models.resnet18(pretrained = True)
    net.fc = nn.Sequential( 
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )"""
    
    # AE + Transformer
    ext = create_model()
    ext.load_state_dict(torch.load(
        #road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
        road_root / "20230928_160620" /'20230928_160625model00166.pth', map_location=device) #AE                               
    )
    
    ext.dec = nn.Sequential( 
        nn.Linear(512, 128, bias=True),
    )
    for param in ext.parameters():
        param.requires_grad = False
    #last_layer = ext.dec
    last_layer = list(ext.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True
    
    #ext = torchvision.models.resnet18(pretrained = True)
    #ext.fc = nn.Linear(512,128, bias = True)
    
    #encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
    #ext = net.to(device)
    ext = ext.to(device)
    #net = nn.TransformerEncoder(encoder_layer, num_layers=6)
    emsize = 128  # 埋め込みベクトルの次元
    d_hid = 512 # nn.TransformerEncoderのフィードフォワードネットワークの次元
    nlayers = 6  # nn.TransformerEncoder内のnn.TransformerEncoderLayerの数
    nhead = 8  # nn.MultiheadAttention内のヘッドの数
    dropout = 0.2  # dropoutの割合
    PE = PositionalEncoding(emsize,dropout)
    net = TransformerModel(emsize, nhead, d_hid, nlayers, dropout)
    net = nn.Sequential(
    net,
    nn.Linear(128, 128, bias=True),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4, bias=True),
    )
    
    print(net)
    net = ExtTrans(ext,net,PE)
    net = net.to(device)
    """net = create_model().to(device)
    num_features = 128
    net = nn.Sequential(
    net,
    nn.Linear(128, 128, bias=True),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128,128, bias=True),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 4, bias=True),
    )
    net = net.to(device)
    print(net)"""
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    #optimizer_ext = torch.optim.RAdam(ext.parameters(), lr=0.0001)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.0001)


    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    #criterion = nn.MSELoss()
    LAMBDA_1 = 0.2
    #LAMBDA_1 = 1.
    LAMBDA_2 = 0.05
    #LAMBDA_2 = 1.
    START_AGE = 0
    END_AGE = 3
    VALIDATION_RATE= 0.1
    
    
    criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE)
    #criterion2 = nn.CrossEntropyLoss() #hard label
    criterion2 = nn.KLDivLoss(reduction='batchmean') # soft label
    tensorboard = SummaryWriter(log_dir='./logs',filename_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    #print(net)

    


    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        train_mean_loss = 0.
        train_variance_loss = 0.
        train_softmax_loss = 0.
        train_mae = 0.
        train_index=0.
        #train_loss_mae = 0.
        for batch, (x, soft_labels, y_true, y_class) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y_true,soft_labels,y_class = x.to(device), y_true.to(device) ,soft_labels.to(device),y_class.to(device)
            
            y_pred = net(x)   # Forward

            #yt_one = torch.from_numpy(OnehotEncording(y_class)).to(device)

            mean_loss, variance_loss = criterion1(y_pred, y_class,device)
            
            #softlabel
            soft_labels = hard_to_soft_labels(y_class,4).to(device)
            #soft_labels = create_softlabel(y_true,4).to(device)
            
            #print(soft_labels)
            softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)
            
            #hard label
            #softmax_loss = criterion2(y_pred, y_class)
            #print(softmax_loss)
            
            #loss = softmax_loss
            loss = mean_loss + variance_loss + softmax_loss
            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            status = np.ones(len(y_true))
            
            #index = concordance_index(y_true.cpu().numpy(), pred, status)  
            
            #print(loss)
            #loss = criterion(y_pred,y_true)
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            # Logging
            train_loss += loss.item() / len(train_loader)
            train_mean_loss += mean_loss / len(train_loader)
            train_variance_loss += variance_loss / len(train_loader)
            train_softmax_loss += softmax_loss / len(train_loader)
            train_mae += mae / len(train_loader)
            
            #train_index += index / len(train_loader)
            
            #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}" .format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                #loss.item()
                loss.item(),softmax_loss,mean_loss,variance_loss
            ), end="")
        print("    train MV: {:3.3}".format(train_loss))
        print('')
        print("    train MAE: {:3.3}".format(train_mae))
        print('')
        #print("    train INDEX: {:3.3}".format(train_index))
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), log_root / f"model{epoch:05}.pth")
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
            loss_mae = 0.
            mean_loss_val = 0.
            variance_loss_val = 0.
            softmax_loss_val = 0.
            valid_mae = 0.
            valid_index = 0.
            for batch, (x, soft_labels, y_true,y_class) in enumerate(valid_loader):
                x, y_true, soft_labels, y_class = x.to(device), y_true.to(device) ,soft_labels.to(device), y_class.to(device)
                
                y_pred = net(x)   # Forward
                
                #y_pred = net(x)  # Prediction
                #yt_one = torch.from_numpy(OnehotEncording(y_class)).to(device)
                mean_loss, variance_loss = criterion1(y_pred, y_class,device)
                
                #softlabel
                soft_labels = hard_to_soft_labels(y_class,4).to(device)
                #soft_labels = create_softlabel(y_true,4).to(device)
                softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)
                
                #hard label
                #oftmax_loss = criterion2(y_pred, y_class)
                
                loss = mean_loss + variance_loss + softmax_loss
                #loss = softmax_loss
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                status = np.ones(len(y_true))
                #index = concordance_index(y_true.cpu().numpy(), pred, status)
                #loss = criterion(y_pred,y_true)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_mae += mae  / len(valid_loader)
                mean_loss_val += mean_loss / len(valid_loader)
                variance_loss_val += variance_loss / len(valid_loader)
                softmax_loss_val += softmax_loss / len(valid_loader)
                #valid_index += index / len(valid_loader)
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(),softmax_loss,mean_loss,variance_loss
                    #loss.item()
                ), end="")
            """for x, y_true in train_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = net(x)  # Prediction
                metrics['train']['loss'] += criterion(y_pred, y_true).item() / len(train_loader)
                # metrics['train']['cmat'] += ConfusionMatrix(y_pred, y_true)"""
        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        print("    valid MV: {:3.3}".format(metrics['valid']['loss']))
        print('')
        print("    valid MAE: {:3.3}".format(valid_mae))
        print('')
        
        #print("    valid INDEX: {:3.3}".format(valid_index))
        
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard'''
        #tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('train_MV', train_loss, epoch)
        tensorboard.add_scalar('train_Mean', train_mean_loss, epoch)
        tensorboard.add_scalar('train_Variance', train_variance_loss, epoch)
        tensorboard.add_scalar('train_Softmax', train_softmax_loss, epoch)
        tensorboard.add_scalar('train_MAE', train_mae, epoch)
        #tensorboard.add_scalar('train_Index', train_index, epoch)
        #tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_Mean', mean_loss_val, epoch)
        tensorboard.add_scalar('valid_Variance', variance_loss_val, epoch)
        tensorboard.add_scalar('valid_Softmax', softmax_loss_val, epoch)
        tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        #tensorboard.add_scalar('valid_Index', valid_index, epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
