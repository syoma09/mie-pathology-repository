#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 2022/04/28, 3OS
- EfficientNet B0
- Layer2
- 35 subjects, 35 images
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score

def main():
    pos_label = 1

    # Subject-wise
    # sbj_y_true = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    # sbj_y_rate = [0.399, 0.000, 0.001, 0.001, 0.653, 0.820, 0.999, 0.998, 0.323, 0.837, 0.995, 0.918, 0.006, 1.000, 0.001, 0.002, 0.997, 0.788, 0.999, 0.986, 0.001, 0.260, 0.446, 0.000, 0.000, 0.002, 0.691, 0.994, 1.000, 0.029, 0.001, 0.007, 0.005, 0.010, 0.710, 0.941, 0.219, 0.993, 0.000, 0.980, 0.005, 0.000, 0.000, 0.866, 0.416, 0.976, 0.722]

    # Subject-wise (Uniq)
    sbj_y_true = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
                  0]
    sbj_y_rate = [0.399, 0.001, 0.998, 0.323, 0.837, 0.995, 0.918, 0.006, 1.000, 0.001, 0.002, 0.997, 0.788, 0.999,
                  0.986, 0.001, 0.260, 0.000, 0.691, 0.994, 0.001, 0.010, 0.710, 0.941, 0.219, 0.993, 0.000, 0.980,
                  0.005, 0.000, 0.000, 0.866, 0.416, 0.976, 0.722]
    sbj_y_rate = [1 - v for v in sbj_y_rate]

    print("Subject-wise:", len(sbj_y_true))
    # print("Image-wise:", len(img_y_true))

    """ROC"""
    # fpr, tpr, thresholds = roc_curve(img_y_true, img_y_rate, pos_label=pos_label)
    # auc = roc_auc_score(img_y_true, img_y_rate)
    # if pos_label == 0:
    #     auc = 1 - auc
    # plt.plot(fpr, tpr, label=f"Image-wise   (AUC={auc:.2})", linestyle='dashed')
    # print(f"pos_label={pos_label}, AUC={auc:.3} @Image-wise")

    fpr, tpr, thresholds = roc_curve(sbj_y_true, sbj_y_rate, pos_label=pos_label)
    auc = roc_auc_score(sbj_y_true, sbj_y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label=f"Subject-wise (AUC={auc:.2})", linestyle='solid')
    print(f"pos_label={pos_label}, AUC={auc:.3} @Subject-wise")

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.axis('square')

    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)

    plt.legend()
    plt.tight_layout()

    plt.savefig('35-35_ROC.jpg', bbox_inches='tight')
    plt.savefig('35-35_ROC.pdf', bbox_inches='tight')
    plt.close()

    """Threshold vs. F-measure"""
    thresholds = list(np.arange(0, 1.0, 0.01))

    for label, (y_true, y_rate) in [
        ("Subject-wise", (sbj_y_true, sbj_y_rate))
    ]:
        f1 = [
            f1_score(y_true=y_true, y_pred=[int(r > th) for r in y_rate])
            for th in thresholds
        ]
        print(f"{label:12} f1-max={max(f1):.3} @th={thresholds[np.argmax(f1)]}")

        plt.plot(thresholds, f1, label=f"{label}")

    plt.ylabel("F-measure")
    plt.xlabel('Threshold')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid()
    plt.tick_params(direction='in')
    plt.legend()

    plt.savefig('35-35_f1th.jpg', bbox_inches='tight')
    plt.savefig('35-35_f1th.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
