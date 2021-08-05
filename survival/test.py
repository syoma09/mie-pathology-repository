#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from cnn.metrics import ConfusionMatrix
from survival import load_annotation, get_dataset_root_path, PatchDataset, create_model


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    cudnn.benchmark = True
print(device)


def evaluate(dataset_root, subjects, model_path):
    data_loader = torch.utils.data.DataLoader(
        PatchDataset(dataset_root, subjects),
        # PatchDataset(dataset_root, annotation['test']),
        batch_size=512, shuffle=True, num_workers=os.cpu_count() // 2
    )

    '''
    モデルの構築
    '''
    model = create_model().to(device)
    model.load_state_dict(torch.load(
        model_path,
        map_location=device
    ))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    # Initialize validation metric values
    metrics = {
        'loss': 0.,
        'cmat': ConfusionMatrix(None, None)
    }
    # Calculate validation metrics
    with torch.no_grad():
        for i, (x, y_true) in enumerate(data_loader):
            print("\r Test ({:5}/{:5})".format(
                i, len(data_loader)
            ), end='')

            y_true = y_true.to(device)
            y_pred = model(x.to(device))    # Prediction

            loss = criterion(y_pred, y_true)  # Calculate validation loss
            # print(loss.item())
            metrics['loss'] += loss.item() / len(data_loader)
            metrics['cmat'] += ConfusionMatrix(y_pred, y_true)
    print("")

    # Console write
    print("    valid loss    : {:3.3}".format(metrics['loss']))
    print("      accuracy    : {:3.3}".format(metrics['cmat'].accuracy))
    print("      f-measure   : {:3.3}".format(metrics['cmat'].f1inv))
    print("      precision   : {:3.3}".format(metrics['cmat'].npv))
    print("      specificity : {:3.3}".format(metrics['cmat'].specificity))
    print("      recall      : {:3.3}".format(metrics['cmat'].tnr))
    print("      Matrix:")
    print(metrics['cmat'])

    return metrics['cmat']


def plot_roc(results, path):
    pos_label = 0
    for dataset in ['train', 'valid', 'test']:
        if dataset not in results:
            continue

        y_true = results[dataset]['true']
        y_rate = results[dataset]['rate']

        fpr, tpr, thresholds = roc_curve(y_true, y_rate, pos_label=pos_label)
        auc = roc_auc_score(y_true, y_rate)
        if pos_label == 0:
            auc = 1 - auc

        plt.plot(fpr, tpr, label=f"{dataset} (AUC={auc:.1})")

    plt.title("ROC")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (FPR)')
    plt.axis('square')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid()
    plt.tick_params(direction='in')

    plt.legend()

    plt.savefig(str(path))
    plt.close()


def main():
    target = '2dfs'

    annotation = load_annotation(Path(
        f"~/workspace/mie-pathology/_data/survival_{target}.csv"
    ).expanduser())
    model_path = Path("~/data/_out/mie-pathology/").expanduser()

    patch_size = 1024, 1024
    stride = 512, 512
    # model_path /= "20210730_131449/model00016.pth"
    # model_path /= "20210803_091002/model00036.pth"
    # model_path /= "20210803_165408/model00007.pth"
    model_path /= "20210804_171325/model00003.pth"

    # Subject
    list_df = {}
    for dataset in ['valid', 'train']:
        if len(annotation[dataset]) == 0:
            continue

        temp = {}
        for name, cls in annotation[dataset]:
            cmat = evaluate(
                dataset_root=get_dataset_root_path(patch_size=patch_size, stride=stride),
                subjects=[(name, cls)],
                model_path=model_path
            )

            temp[name] = {
                "true": cls,
                "pred": np.argmax([cmat.tn + cmat.fn, cmat.tp + cmat.fp]),
                "rate": cmat.accuracy
            }

        list_df[dataset] = pd.DataFrame(temp).transpose()
        print(list_df[dataset])

        # Here is ok, to over-write
        plot_roc(list_df, "roc_test.jpg")

    for key, df in list_df.items():
        print(f"Results on {key} dataset ------------------")
        print(df)

    # # Dataset
    # cmat = evaluate(
    #     dataset_root=get_dataset_root_path(patch_size=patch_size),
    #     subjects=annotation['valid'],
    #     model_path=model_path
    # )


if __name__ == '__main__':
    main()
