#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def plot(y_true, y_rate, name="roc"):
    pos_label = 0

    # fpr, tpr, thresholds = roc_curve(img_y_true, img_y_rate, pos_label=pos_label)
    # auc = roc_auc_score(img_y_true, img_y_rate)
    # if pos_label == 0:
    #     auc = 1 - auc
    # plt.plot(fpr, tpr, label='Image-wise', linestyle='dashed')
    # print(f"pos_label={pos_label}, AUC={auc} @Image-wise")

    fpr, tpr, thresholds = roc_curve(y_true, y_rate, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label=f"AUC={auc:.3}", linestyle='solid')
    print(f"pos_label={pos_label}, AUC={auc}")

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

    plt.savefig(f"{name}.jpg", bbox_inches='tight')
    plt.savefig(f"{name}.pdf", bbox_inches='tight')
    plt.close()


def main():
    print("-- 3OS-test ROC -------------------------")
    df = pd.read_csv("./summary-3OS-test.csv")

    y_true = df["truth"].values
    y_pred = df["pred"].values
    y_rate = df["rate"].values

    print("Metrics:")
    print("  F-measure: {:.3}".format(f1_score(y_true, y_pred)))
    print("  Precision: {:.3}".format(precision_score(y_true, y_pred)))
    print("  Recall   : {:.3}".format(recall_score(y_true, y_pred)))

    plot(y_true, y_rate, name="roc-3os")

    df = pd.read_csv("./summary-3MFS-test.csv")

    print("-- 3MFS-test ROC ------------------------")
    y_true = df["truth"].values
    y_pred = df["pred"].values
    y_rate = df["rate"].values

    print("Metrics:")
    print("  F-measure: {:.3}".format(f1_score(y_true, y_pred)))
    print("  Precision: {:.3}".format(precision_score(y_true, y_pred)))
    print("  Recall   : {:.3}".format(recall_score(y_true, y_pred)))

    plot(y_true, y_rate, name="roc-3mfs")

if __name__ == '__main__':
    main()
