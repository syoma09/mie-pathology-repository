import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import yaml
import math
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import AutoEncoder
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax
from collections import OrderedDict

# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:1'
if torch.cuda.is_available():
    cudnn.benchmark = True
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, subjects):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.paths = []
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
        #print(self.paths[0])
        print(len(self.paths))
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, item):
        """
        :param item:    Index of item
        :return:        Return tuple of (image, label)
                        Label is always "10" <= MetricLearning
        """
    # img = self.data[item, :, :, :].view(3, 32, 32)
        img = Image.open(self.paths[item]).convert('RGB')
        img = self.transform(img)
        #img = torchvision.transforms.functional.to_tensor(img)

        # s0_st34.229508200000005_e0_et34.229508200000005_00000442img.png
        name = self.paths[item].name                # Filename
        label = float(str(name).split('_')[1][2:])  # Survival time

        # Normalize
        #label /= 90.
        '''if(label < 13):
            label_class = 0
        elif(label < 34):
            label_class = 1
        elif(label < 67):
            label_class = 2'''
        if(label < 12):
            label_class = 0
        elif(label < 24):
            label_class = 1
        elif(label < 36):
            label_class = 2
        elif(label < 48):
            label_class = 3
        '''elif(label < 30):
            label_class = 4
        elif(label < 36):
            label_class = 5
        elif(label < 42):
            label_class = 6
        elif(label < 48):
            label_class = 7'''
        '''elif(label < 24):
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
        return img, label, label_class

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

def Mean_Variance_loss(y_pred,y_class):
    N = 4 #Number of classes
    #print("y_pred:",y_pred)
    #print("y_true:",y_true)
    #y_true one-hot-encording
    yt_one = np.zeros((len(y_class),N))
    for i in range (len(y_class)):
        yt_one[i][int(y_class[i])] = 1
    y_train_m = np.zeros(len(y_class)) 
    y_train_v = np.zeros(len(y_class))
    for i in range (len(y_pred)):
        for j in range (N): # class label
            y_train_m[i] += j * y_pred[i][j] # Eq.2
        for j in range (N): # class label
            y_train_v[i] += y_pred[i][j] * pow(j-y_train_m[i],2) # Eq.3
    #print("y_train_m:", y_train_m)
    #print('')
    loss_m_np = np.zeros(len(y_class))
    for i in range (len(y_class)):
        for j in range(N):
            if(yt_one[i][j] == 1):
                loss_m_np[i] = pow(y_train_m[i]-j,2) # Calculate mean loss Eq.4
    #Type transformation numpy to tensor
    loss_m_tensor = torch.tensor((loss_m_np).T, device = device,dtype = torch.float32 ,requires_grad=True)
    y_train_v_tensor = torch.tensor((y_train_v).T, device = device,dtype = torch.float32 ,requires_grad=True)
    loss_m = torch.mean(loss_m_tensor)/2.0
    loss_v = torch.mean(y_train_v_tensor)  # Calculate variance loss Eq.5
    loss_s_tensor = torch.zeros(len(y_class),device = device,dtype = torch.float32 )
    #Caliculate Softmax Loss Eq.6
    for i in range (len(y_class)):
        for j in range(N):
            if(yt_one[i][j] == 1):
                if(-1 * torch.log(y_pred[i][j]).item() == math.inf):
                    print(1)
                    loss_s_tensor[i] = torch.log(y_pred[i][j]+1.0E-8)
                else:
                    loss_s_tensor[i] = torch.log(y_pred[i][j])
    loss_s = torch.mean(loss_s_tensor)
    return loss_s,loss_m,loss_v

def valid_loss(y_pred,y_class):
    N = 4 #Number of classes
    loss = np.zeros(len(y_class))
    for i in range (len(y_pred)):
        for j in range (N):
            loss[i] += y_pred[i][j] * (12 * j + 6)
    loss_tensor = torch.tensor((loss).T, device = device,dtype = torch.float32, requires_grad=True)
    return loss_tensor

class AutoEncoder2(torch.nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
    def forward(self, x):
        x = self.enc(x)
        print(num_flat_features(x))
        #x = x.view(-1,self.num_flat_features(x))
        print(x.size())
        x = self.dec(x)
        print(x.size())
        return x

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

def main():
    # echo $HOME == ~
    src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    # Write dataset on SSD (/mnt/cache/)
    dataset_root = Path("/mnt/cache").expanduser()/ os.environ.get('USER') / 'mie-pathology'
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)
    epochs = 10000
    batch_size = 32     # 64 requires 19 GiB VRAM
    num_workers = os.cpu_count() // 2   # For SMT
    # Load train/valid yaml
    with open(src / "survival_time.yml", "r") as f:
        yml = yaml.safe_load(f)

    # print("PatchDataset")
    # d = PatchDataset(root, yml['train'])
    # d = PatchDataset(root, yml['valid'])
    # print(len(d))
    #
    # print("==PatchDataset")
    # return

    # データ読み込み
    train_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, yml['train']), batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, yml['valid']), batch_size=batch_size,
        num_workers=num_workers
    )
    '''iterator = iter(train_loader)
    x, _ = next(iterator)
    imshow(x)'''
    '''
    モデルの構築
    '''
    net = AutoEncoder.create_model()
    net.load_state_dict(torch.load(
        dataset_root / '20220222_032128model00257.pth', map_location=device)
    )
    #num_features = net.fc.in_features
    # print(num_features)  # 512
    #net.dec = nn.ReLU()
    net.dec = nn.Sequential(
        #nn.Linear(512, 512, bias=True),
        #nn.Linear(512, 512, bias=True),
        nn.Linear(512, 4, bias=True),
        nn.Softmax(dim=1)
        # nn.Sigmoid()
    )
    '''for param in net.parameters():
        param.requires_grad = False
    last_layer = list(net.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True'''
    net.load_state_dict(torch.load(
        dataset_root / '20220317_114715model05299.pth', map_location=device)
        )
    net = net.to(device)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    tensorboard = SummaryWriter(log_dir='./logs')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    print(net)
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss=0.
        train_loss_mae = 0.
        for batch, (x, y_true, y_class) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y_true,y_class = x.to(device), y_true.to(device) ,y_class.to(device)
            y_pred = net(x)   # Forward
            #print("yp:", y_pred.size())
            #print("yt:", y_true)
            answer = Mean_Variance_loss(y_pred,y_class) 
            loss_s = answer[0]
            loss_m = answer[1]
            loss_v = answer[2]
            loss = -1 * loss_s + loss_m + loss_v
            valid_loss_tensor = valid_loss(y_pred,y_class)
            loss_mae = criterion(valid_loss_tensor,y_true)
            #loss = criterion(y_pred,y_true)
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            # Logging
            train_loss += loss.item() / len(train_loader)
            train_loss_mae += loss_mae.item() / len(train_loader)
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")
        print("train_loss",train_loss)
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), dataset_root / f"{model_name}{epoch:05}.pth")
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
            valid_loss_mae = 0.
            for batch, (x, y_true, y_class) in enumerate(valid_loader):
                x, y_true, y_class = x.to(device), y_true.to(device) ,y_class.to(device)
                y_pred = net(x)  # Prediction
                answer = Mean_Variance_loss(y_pred,y_class)
                loss_s = answer[0]
                loss_m = answer[1]
                loss_v = answer[2]
                loss = -1 * loss_s + loss_m + loss_v
                valid_loss_tensor = valid_loss(y_pred,y_class)
                #print(valid_loss_tensor)
                loss_mae = criterion(valid_loss_tensor,y_true)
                #print(loss_mae)
                #loss = criterion(y_pred,y_true)
                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                valid_loss_mae += loss_mae.item() / len(valid_loader)
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
        print("    valid loss: {:3.3}".format(metrics['valid']['loss']))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard'''
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('train_MV', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('valid_MV', metrics['valid']['loss'], epoch)
        tensorboard.add_scalar('train_MAE', train_loss_mae, epoch)
        tensorboard.add_scalar('valid_MAE', valid_loss_mae, epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
