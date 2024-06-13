#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
from pathlib import Path

import torch
from torch.backends import cudnn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from aipatho.svs import TumorMasking
from aipatho.utils.directory import get_cache_dir
from aipatho.dataset import PatchCLDataset, load_annotation, create_dataset
from aipatho.metrics.label import TimeToTime
from aipatho.metrics import InfoNCELoss
from aipatho.model import SimCLR


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(torch.version.cuda)
print("PyTorch version:", torch.__version__)
print("CUDA device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
device = 'cuda:0'
if torch.cuda.is_available():
    cudnn.benchmark = True


# def define_param_groups(model, weight_decay, optimizer_name):
#     def exclude_from_wd_and_adaptation(name):
#         if 'bn' in name:
#             return True
#         if optimizer_name == 'lars' and 'bias' in name:
#             return True
#
#     param_groups = [{
#         'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
#         'weight_decay': weight_decay,
#         'layer_adaptation': True,
#     }, {
#         'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
#         'weight_decay': 0.,
#         'layer_adaptation': False,
#     }]
#     return param_groups


# a lazy way to pass the config file
# class Hparams:
#     def __init__(self):
#         self.epochs = 300 # number of training epochs
#         self.seed = 77777 # randomness seed
#         self.cuda = True # use nvidia gpu
#         self.img_size = 256 #image shape
#         self.save = "./saved_models/" # save checkpoint
#         self.load = False # load pretrained checkpoint
#         self.gradient_accumulation_steps = 5 # gradient accumulation steps
#         self.batch_size = 32
#         self.lr = 3e-4 # for ADAm only
#         self.weight_decay = 1e-6
#         self.embedding_size= 512 # papers value is 128
#         self.temperature = 0.5 # 0.1 or 0.5
#         self.checkpoint_path = './SimCLR_ResNet18.ckpt' # replace checkpoint path here



# class AddProjection(nn.Module):
#     def __init__(self, config, model=None, mlp_dim=512):
#         super(AddProjection, self).__init__()
#
#         embedding_size = config.embedding_size
#         self.backbone = self.default(model, torchvision.models.resnet18(pretrained=False, num_classes=config.embedding_size))
#         mlp_dim = self.default(mlp_dim, self.backbone.fc.in_features)
#         print('Dim MLP input:',mlp_dim)
#         self.backbone.fc = nn.Identity()
#
#         # add mlp projection head
#         self.projection = nn.Sequential(
#             nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
#             # nn.BatchNorm1d(mlp_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=mlp_dim, out_features=embedding_size),
#             # nn.BatchNorm1d(embedding_size),
#         )
#
#     def forward(self, x, return_embedding=False):
#         embedding = self.backbone(x)
#         if return_embedding:
#             return embedding
#         return self.projection(embedding)
#
#     @staticmethod
#     def default(val, def_val):
#         return def_val if val is None else val
#
#
# class SimCLR_pl(pl.LightningModule):
#    def __init__(self, config, model=None, feat_dim=512):
#        super().__init__()
#        self.config = config
#        self.model = AddProjection(config, model=model, mlp_dim=feat_dim)
#
#    def forward(self, x):
#        return self.model(x)


def main():
    patch_size = 256, 256
    stride = 256, 256
    index = 2           # FIXME: No need?
    target = TumorMasking.FULL
    dataset_root = get_cache_dir(
        patch=patch_size,
        stride=stride,
        target=target
    )
    # dataset_root_not = get_cache_dir(
    #     patch=patch_size,
    #     stride=stride,
    #     target=TumorMasking.SEVERE
    # )

    annotation_path = Path(
        "_data/survival_time_cls/20230627_survival_and_nonsurvival.csv"
    ).expanduser()

    # Existing subjects are ignored in the function
    create_dataset(
        src=Path("/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/"),
        dst=dataset_root,
        annotation=annotation_path,
        size=patch_size, stride=stride,
        index=index, region=None, target=target
    )

    # Log, epoch-model output directory
    log_root = Path("~/data/_out/mie-pathology/").expanduser() / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_root.mkdir(parents=True, exist_ok=True)

    # Load annotations
    annotation = load_annotation(annotation_path)
    epochs = 10_000
    batch_size = 32     # 64 requires 19 GiB 
    num_workers = os.cpu_count() // 2   # For SMT
    # train_config = Hparams()

    # net = SimCLR_pl(train_config, model=torchvision.models.resnet18(pretrained=False), feat_dim=512).to(device)
    # net = SimCLR_pl(None, model=torchvision.models.resnet18(pretrained=False), feat_dim=512).to(device)
    net = SimCLR(
        backbone=torchvision.models.resnet18(weights=None, num_classes=512)
    ).to(device)
    # net = torch.nn.DataParallel(net).to(device)

    transform = torchvision.transforms.Compose([
        T.RandomResizedCrop(size=256),
        T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomApply([T.GaussianBlur((3, 3), (0.1, 2.0))], p=0.5),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        # ImageNet stats
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Initialize data loaders
    train_loader = torch.utils.data.DataLoader(
        # torch.utils.data.ConcatDataset([
        #     PatchDataset(dataset_root, annotation['train']),
        #     PatchDataset(dataset_root_not, annotation['train'])
        # ]),
        PatchCLDataset(dataset_root, annotation['train'], transform, TimeToTime()),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        # torch.utils.data.ConcatDataset([
        #     PatchDataset(dataset_root, annotation['valid']),
        #     PatchDataset(dataset_root_not, annotation['valid'])
        # ]),
        PatchCLDataset(dataset_root, annotation['valid'], transform, TimeToTime()),
        batch_size=batch_size, num_workers=num_workers
    )

    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.RAdam(net.parameters(), lr=0.001)

    # criterion = nn.BCELoss()
    # criterion = ContrastiveLoss(temperature=0.5, device=device)
    # criterion = SupConLoss(temperature=0.07)
    criterion = InfoNCELoss(device=device)
    # criterion = nn.CrossEntropyLoss()

    tensorboard = SummaryWriter(log_dir=str(log_root))
    # model_name = "{}model".format(
    #     datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    # )

    for epoch in range(epochs):
        # Initialize metrics logging
        metrics = {
            'train': {
                'loss': 0.,
            }, 'valid': {
                'loss': 0.,
            }
        }

        print(f"Epoch [{epoch:5}/{epochs:5}]:")
        # Switch to training mode
        net.train()

        train_loss = 0.
        for batch, (x1, x2, labels) in enumerate(train_loader):
            # # SimCLR
            # loss = criterion(
            #     proj_1=net(x1.to(device)), proj_2=net(x2.to(device))
            # )

            """SupCon
            images = torch.cat([x1,x2], dim=0)
            images, labels = images.to(device),labels.to(device)
            bsz = labels.shape[0]
            features = net(images)
            # print(features.shape)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # print(f1.shape)
            # print(f2.shape)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            # print(features.shape)
            loss = criterion(features, labels)
            """

            # InfoNCELoss
            loss = criterion(
                x=net(torch.cat([x1, x2], dim=0).to(device)),
                y=torch.cat([labels, labels]).to(device)
            )
            # logits, label = self.info_scs_loss(features, label)
            # nce_lossを使う場合には、ロス関数をクロスエントロピーに
            # scs_lossを使う場合には、ロス関数をMSEに
            # loss = criterion(logits, torch.cat([labels, labels]).to(device))

            # Backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    # Update parameters

            # Logging
            train_loss += loss.item() / len(train_loader)
            print("\r  Batch({:6}/{:6})[{}]: loss={:.4} ".format(
                batch, len(train_loader),
                ('=' * (30 * batch // len(train_loader)) + " " * 30)[:30],
                loss.item()
            ), end="")

        print("train_loss", train_loss)
        print('')
        print('    Saving model...')
        torch.save(net.state_dict(), log_root / f"basecl{epoch:05}.pth")
        
        # Switch to evaluation mode
        net.eval()

        # Calculate validation metrics
        with torch.no_grad():
            for batch, (x1, x2, labels) in enumerate(valid_loader):
                """
                # SimCLR
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
                # 特徴抽出モデルに入力して特徴ベクトルを取得
                original_embedding = net(x1)
                augmented_embedding = net(x2)
                loss = criterion(original_embedding, augmented_embedding)
                """
                """
                # SupCon
                images = torch.cat([x1,x2], dim=0)
                images, labels = images.to(device),labels.to(device)
                bsz = labels.shape[0]
                features = net(images)
                f1, f2 = torch.split(features,[bsz,bsz],dim = 0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = criterion(features, labels)
                """

                # InfoNCELoss
                loss = criterion(
                    x=net(torch.cat([x1, x2], dim=0).to(device)),
                    y=torch.cat([labels, labels]).to(device)
                )

                # InfoSCSLoss
                # images = torch.cat([x1,x2], dim=0)
                # images = images.to(device)
                # features = net(images)
                # logits, label = self.info_scs_loss(features, label)
                # nce_lossを使う場合には、ロス関数をクロスエントロピーに
                # scs_lossを使う場合には、ロス関数をMSEに
                # loss = criterion(logits, torch.cat([labels,labels]).to(device))

                # Logging
                metrics['valid']['loss'] += loss.item() / len(valid_loader)
                print("\r  Validating... ({:6}/{:6})[{}]: loss={:.4}".format(
                    batch, len(valid_loader),
                    ('=' * (30 * batch // len(valid_loader)) + " " * 30)[:30],
                    #loss.item(),loss1,loss2,loss3
                    loss.item()
                ), end="")

        # Console write
        print("    train loss: {:3.3}".format(train_loss))
        print("    valid loss: {:3.3}".format(metrics['valid']['loss']))
        # Write tensorboard
        tensorboard.add_scalar('train_loss', train_loss, epoch)
        tensorboard.add_scalar('valid_loss', metrics['valid']['loss'], epoch)


if __name__ == '__main__':
    main()