import os
from PIL import Image
import csv
import numpy as np

import torch
import torch.utils.data
from torch import nn

import torchvision

class GroupToTensor():
    ''' 画像をまとめてテンソル化するクラス。
    '''

    def __init__(self):
        '''テンソル化する処理を用意'''
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        '''テンソル化をimg_group(リスト)内の各imgに実施
        0から1ではなく、0から255で扱うため、255をかけ算する。
        0から255で扱うのは、学習済みデータの形式に合わせるため
        '''

        return [self.to_tensor(img) for img in img_group]

"""class PathToImg():
    ''' pathをまとめて画像化するクラス。
    '''

    def __init__(self):
        '''テンソル化する処理を用意'''
        self.to_img = Image.open(path).convert('RGB')

    def __call__(self, path_group):
        '''テンソル化をimg_group(リスト)内の各imgに実施
        '''

        return [self.to_img(path) for path in path_group]"""

class VideoTransform():
    """
    動画を画像にした画像ファイルの前処理クラス。学習時と推論時で異なる動作をします。
    動画を画像に分割しているため、分割された画像たちをまとめて前処理する点に注意してください。
    """

    def __init__(self):
        self.data_transform = torchvision.transforms.Compose([
                # DataAugumentation()  # 今回は省略
                #GroupResize(int(resize)),  # 画像をまとめてリサイズ　
                #GroupCenterCrop(crop_size),  # 画像をまとめてセンタークロップ
                GroupToTensor(),  # データをPyTorchのテンソルに
                GroupImgNormalize(),  # データを標準化
                Stack()  # 複数画像をframes次元で結合させる
            ])

    def __call__(self, img_group):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform(img_group)

class GroupImgNormalize():
    ''' 画像をまとめて標準化するクラス。
    '''

    def __init__(self):
        '''標準化する処理を用意'''
        self.normlize = torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        

    def __call__(self, img_group):
        '''標準化をimg_group(リスト)内の各imgに実施'''
        return [self.normlize(img) for img in img_group]

class Stack():
    ''' 画像を一つのテンソルにまとめるクラス。
    '''

    def __call__(self, img_group):
        '''img_groupはtorch.Size([3, 256, 256])を要素とするリスト
        '''
        ret = torch.cat([x.unsqueeze(dim=0) for x in img_group], dim=0)  # frames次元で結合
        # unsqueeze(dim=0)はあらたにframes用の次元を作成しています

        return ret

def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def reshaped_data(data: [(str, float)], seq_len : int = 8):
    """
    data : list of (img_path, label)
    """
    
    #print(x)
    remainder = len(data) % seq_len

    # データをランダムにシャッフル
    #np.random.shuffle(data)

    # 余りがある場合、データの最後から余りの分だけ削除
    if remainder > 0:
        data = data[:-remainder]

    # reshape関数を使って分割
    reshaped_data = [
        data[i:i+seq_len] for i in range(0, len(data), seq_len)  
    ]

    return reshaped_data