import torch
import torch.nn as nn


def estimate_value(y_pred):
    # N = 4   # Number of classes
    a = torch.arange(6, 48, 12, dtype=torch.float32)
    m = nn.Softmax(dim = 1)
    output = m(y_pred).cpu()
    #print(output)
    #print('')
    pred = (output * a).sum(1, keepdim=True).cpu().data.numpy()
    return pred



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
        if d_label_l < labels and labels <= d_label_r:
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


