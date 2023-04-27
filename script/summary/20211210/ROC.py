#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score


def plot(y_true, y_rate, pos_label=1, title: str = "roc"):
    fpr, tpr, thresholds = roc_curve(y_true, y_rate, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label='Subject-wise', linestyle='dashed')
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

    plt.savefig(f"{title}.jpg", bbox_inches='tight')
    plt.savefig(f"{title}.pdf", bbox_inches='tight')
    plt.close()


def main():
    y_true = np.array([
        0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0
    ])


    print("-- Subject-wise (F1, SGD, Death detection) -------------------------")
    plot(
        y_true=y_true,
        y_rate=np.array([0.333, 0.734, 0.424, 0.555, 0.679, 0.562, 0.729, 0.393, 0.650, 0.875, 0.866, 0.989, 0.632, 0.483, 0.646, 0.377, 0.772, 0.860, 0.756, 0.681, 0.505, 0.791, 0.408, 0.675, 0.394, 0.494, 0.092, 0.020, 0.189, 0.571]),
        title="f1-sgd-death"
    )
    print("-- Subject-wise (F1, Adam, Death detection) -------------------------")
    plot(
        y_true=y_true,
        y_rate=np.array([0.396, 0.784, 0.628, 0.681, 0.794, 0.486, 0.898, 0.685, 0.736, 0.798, 0.862, 0.993, 0.765, 0.405, 0.580, 0.511, 0.610, 0.733, 0.736, 0.556, 0.444, 0.908, 0.549, 0.888, 0.459, 0.662, 0.118, 0.031, 0.232, 0.656]),
        title="f1-adam-death"
    )
    print("-- Subject-wise (F1inv, SGD, Death detection) -----------------------")
    plot(
        y_true=y_true,
        y_rate=np.array([0.862, 0.757, 0.881, 0.816, 0.949, 0.749, 0.922, 0.972, 0.817, 0.944, 0.940, 0.926, 0.929, 0.845, 0.862, 0.751, 0.974, 0.861, 0.836, 0.930, 0.681, 0.919, 0.901, 0.894, 0.733, 0.758, 0.346, 0.285, 0.584, 0.752]),
        title="f1inv-sgd-death"
    )
    print("-- Subject-wise (F1inv, Adam, Death detection) ----------------------")
    plot(
        y_true=y_true,
        y_rate=np.array([0.396, 0.784, 0.628, 0.681, 0.794, 0.486, 0.898, 0.838922, 0.804348, 0.906042, 0.827792, 0.993757, 0.846853, 0.55744, 0.715975, 0.684523, 0.71078, 0.888507, 0.899282, 0.687013, 0.445943, 0.907905, 0.54903, 0.887883, 0.458985, 0.661667, 0.118011, 0.030546, 0.232029, 0.655825]),
        title="f1inv-adam-death"
    )


if __name__ == '__main__':
    main()
