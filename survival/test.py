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
        batch_size=128, shuffle=True, num_workers=os.cpu_count() // 2
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


def filter_images(patient, annotation):
    """

    :param patient:
    :param annotation:
    :return:
    """

    return [
        (name, cls)
        for name, cls in annotation
        if name.startswith(patient)
    ]


def main():
    patch_size = 1024, 1024
    stride = 512, 512

    # 3-OS
    # cv, date, epoch = 0, "20220615_143124", 292
    # cv, date, epoch = 1, "20220615_143117", 185
    # cv, date, epoch = 2, "20220615_150709", 5
    # cv, date, epoch = 3, "20220616_110237", 200

    # 3-MFS
    # cv, date, epoch = 0, "20220617_122426", 249
    # cv, date, epoch = 1, "20220616_183431", 45
    # cv, date, epoch = 2, "20220620_132834", 182
    cv, date, epoch = 3, "20220620_132625", 224

    # Load annotation
    annotation = load_annotation(Path(
        # f"~/workspace/mie-pathology/_data/survival_cls2/cv{cv}.csv"
        # f"../_data/20220428_3os/cv{cv}.csv"
        # f"../_data/20220610_3os/cv{cv}.csv"
        f"../_data/20220610_3mfs/cv{cv}.csv"
    ).expanduser())

    model_path = Path(
        # f"~/data/_out/mie-pathology/{date}_w{patch_size[0]}s{stride[0]}cv{cv}_{optim}/model{epoch:05}.pth"
        f"~/data/_out/mie-pathology/{date}/model{epoch:05}.pth"
    ).expanduser()

    """
    Subject-wise experiment
    """
    annotation_patient = {
        dataset: sorted(set([
            (name.split('-')[0], cls)
            for name, cls in values
        ])) for dataset, values in annotation.items()
    }

    # Subject
    list_df = {}
    # for dataset in ['valid']:
    for dataset in ['test']:
        if len(annotation[dataset]) == 0:
            continue
        # print(annotation[dataset])

        temp = {}
        # for name, cls in annotation[dataset]:
        #     subjects = [(name, cls)]
        for name, cls in annotation_patient[dataset]:
            subjects = filter_images(name, annotation[dataset])
            print(subjects)

            cmat = evaluate(
                dataset_root=get_dataset_root_path(patch_size=patch_size, stride=stride),
                subjects=subjects,
                model_path=model_path
            )

            temp[name] = {
                "true": cls,
                "pred": np.argmax([cmat.tn + cmat.fn, cmat.tp + cmat.fp]),
                # Probability of un-survival (label==0)
                "rate": (cmat.tn + cmat.fn) / (cmat.tn + cmat.fn + cmat.tp + cmat.fp)
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
