import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.utils.data
import torchvision
import timm
import math
import random
import numpy as np
import pandas as pd
import sys
import itertools
import matplotlib.pyplot as plt
from AutoEncoder import create_model
from models.attention import Attention
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import OrderedDict
from lifelines.utils import concordance_index
from dataset_path import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from data.svs import save_patches
from AgeEstimation.mean_variance_loss import MeanVarianceLoss
from VAE import VAE
from Unet import Generator
from contrastive_learning import Hparams,SimCLR_pl,AddProjection
from transformer_model import ExtTrans, PositionalEncoding, TransformerModel, features_sort
from create_soft_labels import estimate_value, create_softlabel_tight, hard_to_soft_labels, create_softlabel_survival_time_wise
from three_dimention_dataloader import GroupToTensor, VideoTransform, GroupImgNormalize, Stack, split_list, reshaped_data
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

        #CNN
        for subject, label in annotations:
            self.__dataset += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
            ]

        #Transformer
        """for subject, label in annotations:
            paths = []
            paths += [
                (path, label)   # Same label for one subject
                for path in (root / subject).iterdir()
            ]
            #print(paths)
            #random.shuffle(paths)
            data = reshaped_data(paths)
            #print(data)
            self.__dataset += data"""
        #print((self.__dataset))
        #self.__dataset = list(itertools.chain.from_iterable(self.__dataset))
        random.shuffle(self.__dataset)
        #print((self.__dataset))
        #print(len(self.__dataset))
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

        """print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([l for _, l in self.__dataset if l <= 11]))
        print('  # of 1  :', len([l for _, l in self.__dataset if (11 < l) & (l <= 22)]))
        print('  # of 2  :', len([l for _, l in self.__dataset if (22 < l) & (l <= 33)]))
        print('  # of 3  :', len([l for _, l in self.__dataset if (33 < l) & (l <= 44)]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _ in self.__dataset])))
        """

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
        #self.l = self.__dataset[item]
        #print(self.l)
        #path, label = self.l[item]
        
        # Transformer
        """path_group, [label, *_] = self.pull_group(item)
        #print(type(path_group[0]))
        img_group = self.pathtoimg(path_group)
        group_transform = VideoTransform()
        img_group = group_transform(img_group)
        #label = label_group[0]"""
        
        # CNN
        path, label = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        #label /= 90.
        
        if(label < 11):
            label_class = 0
        elif(label < 22):
            label_class = 1
        elif(label < 33):
            label_class = 2
        elif(label < 44):
            label_class = 3

        # Tensor
        label = torch.tensor(label, dtype=torch.float)
        
        #softlabel
        num_classes = 4
        #soft_labels = hard_to_soft_labels(label_class,num_classes)#basic
        #soft_labels = create_softlabel_tight(label,num_classes)#tight
        soft_labels = create_softlabel_survival_time_wise(label,num_classes)#survivaltime_wise
        
        return img, soft_labels, label, label_class #CNN
        #return img_group, soft_labels, label, label_class #Transformer

    def pull_group(self,item :int)->([str],[float]):

        return (
           [path for path, _  in self.__dataset[item]],
           [label for _, label  in self.__dataset[item]] 
        )
    
    def pathtoimg(self,path_group):
        return [Image.open(path).convert('RGB') for path in path_group] 


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


def create_dataset(
        src: Path, dst: Path,
        annotation: Path,
        size, stride,
        index: int = None, region: int = None
):
    # Load annotation
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
        #print(index)
        args.append((path_svs, path_xml, base, size, stride, resize))
        # # Serial execution
        # save_patches(path_svs, path_xml, base, size=size, stride=stride)

    # Approx., 1 thread use 20GB
    #mem_total = 
    #n_jobs = int(mem_total / 20)
    n_jobs = 8
    print(f'Process in {n_jobs} threads.')
    #print(index)
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
    road_root = Path("~/data/_out/mie-pathology/").expanduser()
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)
    annotation_path = Path(
        #"../_data/survival_time_cls/20220726_cls/cv0.csv"
        #"../_data/survival_time_cls/20220726_cls/cv1.csv"
        #"../_data/survival_time_cls/20220726_cls/cv2.csv"
        "../_data/survival_time_cls/20220726_cls.csv"
        #"../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        #"../_data/survival_time_cls/fake_data.csv"
    ).expanduser()
    # Create dataset if not exists
    """if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)"""
    
    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None
    )
    
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
        PatchDataset(dataset_root, annotation['train'],flag), batch_size=batch_size,shuffle = True,
        num_workers=num_workers,drop_last = True
    )

    flag = 1
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'],flag), batch_size=batch_size,
        num_workers=num_workers,drop_last = True
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
    #AE
    net = create_model() 
    
    net.load_state_dict(torch.load(
        #road_root / "20230919_175330" /'20230919_175350model00125.pth', map_location=device) #AE
        road_root / "20230928_160620" /'20230928_160625model00166.pth', map_location=device) #AE                                
    )

    net.dec = nn.Sequential( #AE
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4, bias=True),
    )

    # contrastive
    """train_config = Hparams()
    net = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512)
    
    net.model.projection = nn.Sequential(
    #net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        #nn.BatchNorm1d(512),
        nn.ReLU(),
        #nn.Dropout(0.5),
        nn.Linear(512, 512, bias=True),
    )
    
    net.load_state_dict(torch.load(
        road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive                                
    )
    
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    #print(net.model.projection)
    
    net.model.projection = nn.Sequential( #CL
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
    
    
    # ResNet
    
    """net = torchvision.models.resnet18(pretrained = True)
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
    )
    """
    
    
    #Selfsupervised + Transformer
    """ext = create_model()
    # contrastive
    #train_config = Hparams()
    #ext = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512)
    ext.load_state_dict(torch.load(
        #road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
        road_root / "20230928_160620" /'20230928_160625model00166.pth', map_location=device) #AE                               
    )
    
    # AE
    ext.dec = nn.Sequential( 
        #nn.Linear(512, 128, bias=True),
    )

    for param in ext.parameters():
        param.requires_grad = False
    #last_layer = ext.dec
    last_layer = list(ext.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True

    ext = ext.to(device)
    
    model_name = "SelfAttention_4096_noExit"
    num_layers =  12
    d_model = 512
    dim = 512
    #embed_dim = num_layers * dim
    embed_dim = 512
    dff = 4096 * 2
    num_heads = 4
    dropout_rate = 0.5
    learning_rate = 0.001
    est = Attention(d_model, embed_dim, num_heads, num_layers, dropout=dropout_rate,
                      dff=dff, device=device).to(device)
    
    PE = PositionalEncoding(d_model,dropout_rate)
    net = ExtTrans(ext,est,PE)
    """
    
    net = net.to(device)
    

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    #optimizer_ext = torch.optim.RAdam(ext.parameters(), lr=0.0001)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.001)

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
    print(net)

    seq_len = 8
    num_classes = 4

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
        train_loss_mae = 0.
        for batch, (x, soft_labels, y_true, y_class) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y_true,soft_labels,y_class = x.to(device), y_true.to(device) ,soft_labels.to(device),y_class.to(device)
            
            """
            #Transformer
            sort_features_list = []
            sort_clusters_list = []
            for i in range(len(x)):
                #print(x[i,:,:,:].shape)
                feature = ext(x[i,:,:,:])
                #print(feature.shape)
                sort_features,sort_clusters = features_sort(feature)
                #feature_list.append(feature)
                sort_features_list.append(sort_features)
                sort_clusters_list.append(sort_clusters)
            combined_feature = torch.stack(sort_features_list,dim=0)
            
            y_pred = net(combined_feature,sort_clusters_list)   # Forward
            """
            y_pred = net(x)

            mean_loss, variance_loss = criterion1(y_pred, y_class, device)
            
            #soft label
            #print(soft_labels)
            softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)
            
            #hard label
            #softmax_loss = criterion2(y_pred, y_class)
            #print(softmax_loss)
            
            #loss = softmax_loss
            loss = mean_loss + variance_loss +  softmax_loss
            pred = estimate_value(y_pred)
            pred = np.squeeze(pred)
            mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
            #print('')
            #print(mae)
            #print('')
            status = np.ones(len(y_true))
            
            index = concordance_index(y_true.cpu().numpy(), pred, status)  
            
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
            
            train_index += index / len(train_loader)
            
            #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}" .format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                #loss.item()
                loss.item(),softmax_loss,mean_loss,variance_loss
            ), end="")
            print(f" {mae:.3}", end="")
        print("    train MV: {:3.3}".format(train_loss))
        print('')
        print("    train MAE: {:3.3}".format(train_mae))
        print('')
        print("    train INDEX: {:3.3}".format(train_index))
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
                #Transformer
                """# テンソルを格納するリスト
                sort_features_list = []
                sort_clusters_list = []
                for i in range(len(x)):
                    feature = ext(x[i,:,:,:])
                    #print(feature.shape)
                    sort_features,sort_clusters = features_sort(feature)
                    #feature_list.append(feature)
                    sort_features_list.append(sort_features)
                    sort_clusters_list.append(sort_clusters)
                combined_feature = torch.cat(sort_features_list)
                combined_feature = torch.reshape(combined_feature,(batch_size,8,d_model))
                #print(combined_feature.shape)  
                y_pred = net(combined_feature,sort_clusters_list)   # Forward
                #y_pred = net(combined_feature)   # Forward
                """

                y_pred = net(x)  # Prediction
                #yt_one = torch.from_numpy(OnehotEncording(y_class)).to(device)
                mean_loss, variance_loss = criterion1(y_pred, y_class,device)
                
                #soft label
                softmax_loss = criterion2(torch.log_softmax(y_pred, dim=1), soft_labels)

                #hard label
                #softmax_loss = criterion2(y_pred, y_class)
                
                loss = mean_loss + variance_loss +  softmax_loss
                #loss = softmax_loss
                
                pred = estimate_value(y_pred)
                pred = np.squeeze(pred)
                mae = np.absolute(pred - y_true.cpu().data.numpy()).mean()
                status = np.ones(len(y_true))
                index = concordance_index(y_true.cpu().numpy(), pred, status)
                #loss = criterion(y_pred,y_true)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_mae += mae  / len(valid_loader)
                mean_loss_val += mean_loss / len(valid_loader)
                variance_loss_val += variance_loss / len(valid_loader)
                softmax_loss_val += softmax_loss / len(valid_loader)
                valid_index += index / len(valid_loader)
                # metrics['valid']['cmat'] += ConfusionMatrix(y_pred, y_true)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(),softmax_loss,mean_loss,variance_loss
                    #loss.item()
                ), end="")
        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        print("    valid MV: {:3.3}".format(metrics['valid']['loss']))
        print('')
        print("    valid MAE: {:3.3}".format(valid_mae))
        print('')
        print("    valid INDEX: {:3.3}".format(valid_index))
        
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
        tensorboard.add_scalar('train_Index', train_index, epoch)
        #tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_Mean', mean_loss_val, epoch)
        tensorboard.add_scalar('valid_Variance', variance_loss_val, epoch)
        tensorboard.add_scalar('valid_Softmax', softmax_loss_val, epoch)
        tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        tensorboard.add_scalar('valid_Index', valid_index, epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
