import torch
import torch.nn as nn
import numpy as np


def estimate_value(y_pred):
    N = 4 #Number of classes
    a = torch.arange(6,48,12, dtype=torch.float32)
    m = nn.Softmax(dim = 1)
    output = m(y_pred).cpu()
    #print(output)
    #print('')
    pred = (output * a).sum(1, keepdim=True).cpu().data.numpy()
    return pred

def OnehotEncording(y_class):
    #One-hot encording
    N = 4
    yt_one = np.zeros((len(y_class),N))
    for i in range (len(y_class)):
        yt_one[i][int(y_class[i])] = 1
    return yt_one

def hard_to_soft_labels(hard_labels, num_classes):
    """
    ハードラベルをソフトラベルに変換する関数    Parameters:
        hard_labels (torch.Tensor): バッチサイズ分のハードラベル (サイズ: [batch_size])
        num_classes (int): クラスの数    Returns:
        torch.Tensor: ソフトラベル (サイズ: [batch_size, num_classes])
    """
    # バッチサイズ取得
    batch_size = hard_labels.size(0)    # 正解クラスに対応するインデックスを取得
    target_indices = hard_labels.view(-1, 1)    # ソフトラベルを作成
    soft_labels = torch.zeros(batch_size, num_classes)    
    # 正規分布に従う確率を計算してソフトラベルに設定
    for i in range(batch_size):
        k = 1.0
        target_class = target_indices[i, 0].item()
        target_class = torch.tensor(target_class)
        sum_class = 0.
        for c in range(num_classes):
            sum_class += torch.sum(torch.exp((-k * (target_class - c) ** 2)))
        prob = [torch.exp(-k *(target_class - c) ** 2) / sum_class for c in range(num_classes)]
        
        soft_labels[i] = torch.tensor(prob)
        #soft_labels[i] = soft_labels[i] / torch.sum(soft_labels[i])
    return soft_labels


def create_softlabel_tight(labels, num_classes):
    soft_labels = torch.zeros(num_classes)   
    for j in range(num_classes*2):
        if(6 * j < labels and labels <= 6.0 * (j+1)):
            #print(j)
            if(j % 2 == 0 or j == 7):
                soft_labels[int(j//2)] = 1.0
            else:
                #print(j)
                #print(j//2)
                #print(int(j+2-1//2))
                soft_labels[int(j//2)] = 0.5
                soft_labels[int(j+2-1)//2] = 0.5
    #soft_labels = soft_labels.T
    return soft_labels

def create_softlabel_survival_time_wise(labels, num_classes):
    soft_labels = torch.zeros(num_classes)   
    for j in range(num_classes-1):
        k = j+1
        d_label_l = 12.0 * j + 6.0
        d_label_r = 12.0 * k + 6.0
        if(d_label_l < labels and labels <= d_label_r):
            left = labels - d_label_l
            right = d_label_r - labels 
            left_ratio = right / (left + right)
            right_ratio = left / (left + right)
            soft_labels[j] = left_ratio
            soft_labels[k] = right_ratio
                
        elif(labels <= 6.0):
            soft_labels[0] = 1.0
        elif(42.0 < labels):
            soft_labels[num_classes-1] = 1.0
    return soft_labels


