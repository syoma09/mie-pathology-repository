import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.utils.data
import torchvision
#import yaml
import math
import random
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from scipy.special import softmax
from collections import OrderedDict
from lifelines.utils import concordance_index
from function import load_annotation, get_dataset_root_path, get_dataset_root_not_path
from data.svs import save_patches
from aipatho.metrics import MeanVarianceLoss
from contrastive_learning import Hparams,SimCLR_pl, AddProjection

# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations,flag):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.__dataset = []
        paths = []
        for subject, label ,status in annotations:
            paths += [
                (path, label, status)   # Same label for one subject
                for path in (root / subject).iterdir()
        ]
        
        
        #self.__dataset += random.sample(paths,1000)
        
        # equal survival and non-survival
        survival_dataset = [item for item in paths if item[2] == 0]
        non_survival_dataset = [item for item in paths if item[2] == 1]
        
        #self.__dataset += random.sample(survival_dataset,15000)
        #self.__dataset += random.sample(non_survival_dataset,15000)
        
        if(len(survival_dataset) > len(non_survival_dataset) and flag == 0):
            self.__dataset += random.sample(survival_dataset,len(non_survival_dataset))
        
        elif(flag == 1):
            self.__dataset += non_survival_dataset
        
        elif(flag == 2):
            if(len(survival_dataset) > len(non_survival_dataset)):
                self.__dataset += random.sample(survival_dataset,len(non_survival_dataset))
                self.__dataset += non_survival_dataset
        
            elif(len(survival_dataset) < len(non_survival_dataset)):
                self.__dataset += random.sample(non_survival_dataset,len(survival_dataset))
                self.__dataset += survival_dataset   



        """if(len(survival_dataset) > len(non_survival_dataset)):
            self.__dataset += random.sample(survival_dataset,len(non_survival_dataset))
            self.__dataset += non_survival_dataset
        
        elif(len(survival_dataset) < len(non_survival_dataset)):
            self.__dataset += random.sample(non_survival_dataset,len(survival_dataset))
            self.__dataset += survival_dataset"""
        #self.__dataset += non_survival_dataset
        
        
        # Random shuffle
        random.shuffle(self.__dataset)
        # reduce_pathces = True
        # if reduce_pathces is True:
        #     data_num = len(self.__dataset) // 5
        #     self.__dataset = self.__dataset[:data_num]

        # self.__num_class = len(set(label for _, label in self.__dataset))
        self.__num_class = 4
        # self.__dataset = self.__dataset[:512]

        print('PatchDataset')
        print('  # patch :', len(self.__dataset))
        print('  # of 0  :', len([s for _, _, s in self.__dataset if s == 0]))
        print('  # of 1  :', len([s for _, _, s in self.__dataset if s == 1]))
        print('  subjects:', sorted(set([str(s).split('/')[-2] for s, _, _ in self.__dataset])))

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
        path, label, status = self.__dataset[item]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        #name = self.paths[item].name                # Filename
        #label = float(str(name).split('_')[1][2:])  # Survival time
        
        # Tensor
        label = torch.tensor(label, dtype=torch.float)
        status = torch.tensor(status, dtype=torch.float)
        return img, label, status


def cox_loss(preds,labels,status):
    labels = labels.unsqueeze(1)
    status = status.unsqueeze(1)

    mask = torch.ones(labels.shape[0],labels.shape[0]).to(device)

    mask[(labels.T - labels)>0] = 0


    log_loss = torch.exp(preds)*mask
    log_loss = torch.sum(log_loss,dim = 0)
    log_loss = torch.log(log_loss).reshape(-1,1)
    log_loss = -torch.sum((preds-log_loss)*status)

    return log_loss

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
        #"../_data/survival_time_cls/20220726_cls.csv"
        "../_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
        #"../_data/survival_time_cls/fake_data.csv"
    ).expanduser()
    # Create dataset if not exists
    """if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)"""
    # Existing subjects are ignored in the function
    
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
    num_workers = os.cpu_count() // 8   # For SMT
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
    """train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    """
    
    flag = 0
    train_survival = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'],flag), batch_size=batch_size, shuffle=True,num_workers=num_workers
    )
    
    flag = 1
    train_non_survival = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['train'],flag), batch_size=batch_size, shuffle=True,num_workers=num_workers
    )
    
    """train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(train_dataset), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )"""
    
    flag = 2
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, annotation['valid'],flag), batch_size=batch_size,
        num_workers=num_workers
    )

    """
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
    
    train_config = Hparams()
    net = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512)
    #net = AddProjection(train_config, model=torchvision.models.resnet18(pretrained=False), mlp_dim=512)
    #net = torchvision.models.resnet18(pretrained=False)
    #net = torchvision.models.vgg16(pretrained = True)
    """num_features = net.fc.in_features
    print(num_features)
    net.fc = nn.Sequential(
        nn.Linear(num_features, 512, bias=True),
        #nn.ReLU(),
        nn.Linear(512, 512, bias=True),
        #nn.ReLU(),
        nn.Linear(512, 2, bias=True),
        #nn.Softmax(dim=1)
        #nn.Sigmoid()
    )"""
    #print(net)
    net.load_state_dict(torch.load(
        #road_root / "20221220_184624" /'', map_location=device) #p=1024,s=512,b=32
        #road_root / "20230309_182905" /'model00520.pth', map_location=device) #f1 best
        #road_root / "20230530_205222" /'20230530_205224model00020.pth', map_location=device) #U-net 
        #road_root / "20230516_162742" /'model00020.pth', map_location=device) #U-net 
        #road_root / "20230712_131831" /'20230712_131837model00140.pth', map_location=device)
        road_root / "20230713_145600" /'20230713_145606model00055.pth', map_location=device) #Contrstive
        #road_root / "20230725_173524" /'20230725_190953model00092.pth', map_location=device)    
    )
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    #print(net.model.projection)
    net.model.projection = nn.Sequential(
    #net.fc = nn.Sequential( 
    #net.dec = nn.Sequential(
        nn.Linear(512, 512, bias=True),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1, bias=True),
    )
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
    
    """for param in net.model.parameters():
        param.requires_grad = False
    #print(net.model.backbone.conv1.bias)
    last_layer = net.model.projection
    #last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True"""
    
    net = net.to(device)
    print(net)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.00001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 10)
    tensorboard = SummaryWriter(log_dir='./logs_cox',filename_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    #print(net)
    
    num_selected_batches = 964 // 2
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        train_index=0.
        # selected_batchesを初期化
        train_loader = []

        for batch_idx, (data, labels, status) in enumerate(train_survival):
            if batch_idx < num_selected_batches:
                print(batch_idx)
                train_loader.append((data, labels, status))
            else:
                break
        batch_idx = 0
        for batch_idx, (data, labels, status) in enumerate(train_non_survival):
            if batch_idx < num_selected_batches:
                print(batch_idx)
                train_loader.append((data, labels, status))
            else:
                break
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
        for batch, (x, y_true, status) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y_true, status = x.to(device), y_true.to(device) , status.to(device)
            y_pred = net(x)   # Forward
            #y_pred = torch.exp(y_pred)
            #y_pred = m(y_pred)
            #print(y_pred.T)
            print(status)
            loss = cox_loss(y_pred, y_true, status)
            index = concordance_index(y_true.cpu().numpy(), -y_pred.cpu().detach().numpy(), status.cpu().numpy())  # Note the negative sign for predictions
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            #optimizer.step()    # Update parameters
            # Logging
            train_loss += loss.item() / len(train_loader)
            train_index += index / len(train_loader)
            
            #train_mae += mae / len(train_loader)
            #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} c_index={:.4}" .format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item(),index
            ), end="")
        print("    train Cox: {:3.3}".format(train_loss))
        print("    train index: {:3.3}".format(train_index))
        #scheduler.step()
        #print('epoch:{}, lr:{:.7}'.format(epoch, scheduler.get_last_lr()[0])) 
        #print("    train MAE: {:3.3}".format(train_mae))
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
            }, 'valid': {
                'loss': 0.,
            }
        }
        # Calculate validation metrics
        with torch.no_grad():
            valid_mae = 0.
            valid_index=0.
            for batch, (x, y_true, status) in enumerate(valid_loader):
                x, y_true, status = x.to(device), y_true.to(device) ,status.to(device)
                y_pred = net(x)  # Prediction
                #y_pred = torch.exp(y_pred)
                #y_pred = m(y_pred)
                loss = cox_loss(y_pred,y_true,status)
                index = concordance_index(y_true.cpu().numpy(), -y_pred.detach().cpu().numpy(), status.cpu().numpy())  # Note the negative sign for predictions
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_index += index / len(valid_loader)
                #print("\r  Validating... ({:6}/{:6})[{}]".format(
                print("\r  Batch({:6}/{:6})[{}]: loss={:.4} c_index={:.4}".format(
                #print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    loss.item(),index
                ), end="")
        # # Console write
        # print("    train loss: {:3.3}".format(metrics['train']['loss']))
        # print("          acc : {:3.3}".format(metrics['train']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['train']['cmat'].f1()))
        print("    valid Cox: {:3.3}".format(metrics['valid']['loss']))
        print("    valid index: {:3.3}".format(valid_index))
        #print("    valid MAE: {:3.3}".format(valid_mae))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard'''
        #tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('train_cox', train_loss, epoch)
        tensorboard.add_scalar('train_index', train_index, epoch)
        #tensorboard.add_scalar('train_MAE', train_mae, epoch)
        #tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_cox', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_index', valid_index, epoch)
        #tensorboard.add_scalar('valid_MAE', valid_mae, epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
