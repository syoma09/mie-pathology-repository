import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import yaml
import math
import numpy
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
from PIL import ImageFile
from cnn.metrics import ConfusionMatrix
from scipy.special import softmax

# To avoid "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0'
if torch.cuda.is_available():
    cudnn.benchmark = True
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, root, subjects):
        super(PatchDataset, self).__init__()
        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(0.5, 0.5)
        ])
        self.paths = []
        for subject in subjects:
            self.paths += list((root / subject).iterdir())
        #print(self.paths[0])
        #print(len(self.paths)
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
        # classification
        if(label < 10):
            label = 0
        elif(label < 31):
            label = 1
        elif(label < 67):
            label = 2
            # Tensor
        #label = torch.tensor(label, dtype=torch.float)
        return img, label

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
def Mean_Variance_loss(y_pred,y_true):
    N = 3 #Number of classes
    #print("y_pred:",y_pred)
    #print("y_true:",y_true)
    #y_true one-hot-encording
    yt_one = numpy.zeros((len(y_true),N))
    for i in range (len(y_true)):
        yt_one[i][int(y_true[i])] = 1
    y_train_m = numpy.zeros(len(y_true)) 
    y_train_v = numpy.zeros(len(y_true))
    for i in range (len(y_pred)):
        for j in range (N): # class label
            y_train_m[i] += j * y_pred[i][j] # Eq.2
        for j in range (N): # class label
            y_train_v[i] += y_pred[i][j] * pow(j-y_train_m[i],2) # Eq.3
    #print("y_train_m:", y_train_m)
    #print('')
    loss_m_np = numpy.zeros(len(y_true))
    for i in range (len(y_true)):
        for j in range(N):
            if(yt_one[i][j] == 1):
                loss_m_np[i] = pow(y_train_m[i]-j,2) # Calculate training  mean loss Eq.4
    #Type transformation numpy to tensor
    loss_m_tensor = torch.tensor((loss_m_np).T, device = device,dtype = torch.float32 ,requires_grad=True)
    y_train_v_tensor = torch.tensor((y_train_v).T, device = device,dtype = torch.float32 ,requires_grad=True)
    loss_m = torch.mean(loss_m_tensor)/2.0
    loss_v = torch.mean(y_train_v_tensor)  # Calculate training varianceloss Eq.5
    loss_s_tensor = torch.zeros(len(y_true),device = device,dtype = torch.float32 )
    for i in range (len(y_true)):
        for j in range(N):
            if(yt_one[i][j] == 1):
                loss_s_tensor[i] = torch.log(y_pred[i][j]) #Caliculate Softmax Loss Eq.6
    loss_s = torch.mean(loss_s_tensor)
    return loss_s,loss_m,loss_v


def main():
    # echo $HOME == ~
    src = Path("~/root/workspace/mie-pathology/_data/").expanduser()
    # Write dataset on SSD (/mnt/cache/)
    dataset_root = Path('/mnt/cache')/ os.environ.get('USER') / 'mie-pathology'
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True,exist_ok=True)
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

    '''
    モデルの構築
    '''
    # model = torchvision.models.resnet152(pretrained=False)
    #net = torchvision.models.resnet152(pretrained=True)
    net = torchvision.models.resnet50(pretrained=True)
    # print(model)
    # Replace FC layer
    num_features = net.fc.in_features
        # print(num_features)  # 512
    net.fc = nn.Sequential(
        nn.Linear(num_features, 3, bias=True),
        nn.Softmax(dim = 1)
        )
    net = net.to(device)
    # net = torch.nn.DataParallel(net).to(device)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()

    criterion = nn.MSELoss()
    tensorboard = SummaryWriter(log_dir='./logss')
    model_name = "{}model".format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    for epoch in range(epochs):
        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()
        train_loss = 0.
        for batch, (x, y_true) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y_true = x.to(device), y_true.to(device)
            y_pred = net(x)   # Forward
            #print("yp:", y_pred)
            #print("yt:", y_true)
            answer = Mean_Variance_loss(y_pred,y_true) 
            loss_s = answer[0]
            loss_m = answer[1]
            loss_v = answer[2]
            loss = -1 * loss_s + 0.5 * loss_m + 0.5 * loss_v
            # Backward propagation
            loss.backward()
            optimizer.step()    # Update parameters
            # Logging
            train_loss += (-1 * loss_s.item() + loss_m.item()  + loss_v.item()) / len(train_loader) 
            print("\r  Batch({:6}/{:6})[{}]: loss_s={:.4} loss_m={:.4} loss_v={:.4}".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss_s.item(),loss_m.item(),loss_v.item()
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
        valid_loss = 0.
        with torch.no_grad():
            for x, y_true in valid_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = net(x)  # Prediction
                answer = Mean_Variance_loss(y_pred,y_true)
                loss_s = answer[0]
                loss_m = answer[1]
                loss_v = answer[2]
                # Logging
                metrics['valid']['loss'] += (-1 * loss_s.item() + loss_m.item() + loss_v.item()) / len(valid_loader)
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
        # print("    valid loss: {:3.3}".format(metrics['valid']['loss']))
        # print("          acc : {:3.3}".format(metrics['valid']['cmat'].accuracy()))
        # print("          f1  : {:3.3}".format(metrics['valid']['cmat'].f1()))
        # print("        Matrix:")
        # print(metrics['valid']['cmat'])
        # Write tensorboard
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)
        # tensorboard.add_scalar('valid_acc', metrics['valid']['cmat'].accuracy(), epoch)
        # tensorboard.add_scalar('valid_f1', metrics['valid']['cmat'].f1(), epoch)
if __name__ == '__main__':
    main()
