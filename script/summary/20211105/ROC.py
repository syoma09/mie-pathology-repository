#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score


def roc(
    img_y_true, img_y_rate,
    sbj_y_true, sbj_y_rate
):
    pos_label = 0

    fpr, tpr, thresholds = roc_curve(img_y_true, img_y_rate, pos_label=pos_label)
    auc = roc_auc_score(img_y_true, img_y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label='Image-wise', linestyle='dashed')
    print(f"pos_label={pos_label}, AUC={auc} @Image-wise")

    fpr, tpr, thresholds = roc_curve(sbj_y_true, sbj_y_rate, pos_label=pos_label)
    auc = roc_auc_score(sbj_y_true, sbj_y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label='Subject-wise', linestyle='solid')
    print(f"pos_label={pos_label}, AUC={auc} @Subject-wise")

    # plt.title("ROC")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.axis('square')

    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)

    # plt.grid()
    # plt.tick_params(direction='in')
    plt.legend()

    plt.tight_layout()

    plt.savefig('roc.jpg', bbox_inches='tight')
    plt.savefig('roc.pdf', bbox_inches='tight')
    plt.close()


def th_shift(y_true, y_rate):
    thresholds = list(np.arange(0, 1.0, 0.01))

    acc = [
        accuracy_score(y_true=y_true, y_pred=[int(r < th) for r in y_rate])
        for th in thresholds
    ]
    f1 = [
        f1_score(y_true=y_true, y_pred=[int(r < th) for r in y_rate])
        for th in thresholds
    ]

    plt.plot(thresholds, acc, label=f"Accuracy")
    plt.plot(thresholds, f1, label=f"F-measure")

    plt.xlabel('Threshold')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid()
    plt.tick_params(direction='in')

    plt.legend()
    # plt.show()

    plt.savefig("th_vs_acc.jpg")
    plt.savefig("th_vs_acc.pdf")
    # files.download("th_vs_acc.pdf")
    plt.close()


def main():
    pos_label = 0

    # Image-wise
    img_y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    img_y_rate = [0.414561, 0.403481, 0.675039, 0.467179, 0.9709, 0.989111, 0.435903, 0.532154, 0.464629, 0.612222, 0.392925, 0.435448, 0.542788, 0.57797, 0.209383, 0.807283, 0.473965, 0.900045, 0.160093, 0.079565, 0.157539, 0.208111, 0.520915, 0.749471, 0.419294, 0.104213, 0.820167, 0.074093, 0.205218, 0.823162, 0.373868, 0.34059, 0.396064, 0.370576, 0.791882, 0.896873, 0.909179, 0.68384, 0.503285, 0.53601, 0.708947, 0.74981, 0.982207, 0.93514, 0.929063, 0.038952, 0.062093, 0.002267, 0.053787, 0.254684, 0.641588, 0.48337, 0.214377, 0.400489, 0.868514, 0.360824, 0.739893, 0.809604, 0.937016, 0.904391, 0.639877]

    # Subject-wise
    sbj_y_true = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    sbj_y_rate = [0.491924, 0.807283, 0.36548, 0.45515, 0.486401, 0.497762, 0.976818, 0.131879, 0.347585, 0.341919, 0.560649, 0.823162, 0.692825, 0.34059, 0.846061, 0.695301, 0.634954, 0.955249, 0.929063, 0.389332, 0.373868, 0.904391, 0.568642, 0.860554, 0.639877, 0.617166, 0.050207, 0.026333, 0.214377, 0.438265]

    # y_true, y_rate = img_y_true, img_y_rate
    y_true, y_rate = sbj_y_true, sbj_y_rate

    print(len(y_true), len(y_rate))

    roc(img_y_true, img_y_rate, sbj_y_true, sbj_y_rate)
    # th_shift(img_y_true, img_y_rate)
    th_shift(sbj_y_true, sbj_y_rate)

if __name__ == '__main__':
    main()
