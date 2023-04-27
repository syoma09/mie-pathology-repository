#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 2021/xx/xx, xOS
- Inception v3
- Layer1
- 26 subjects, 47 images
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score


def main():
    pos_label = 1

    # Image-wise
    img_y_true = [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    img_y_rate = [0.487, 0.676, 0.373, 0.225, 0.332, 0.314, 0.953, 0.942, 0.183, 0.357, 0.643, 0.826, 0.97, 0.951, 0.369, 0.494, 0.864, 0.976, 0.975, 0.429, 0.481, 0.694, 0.983, 0.365, 0.325, 0.64, 0.883, 0.822, 0.673, 0.683, 0.909, 0.996, 0.97, 0.941, 0.95, 0.506, 0.42, 0.269, 0.363, 0.929, 0.864, 0.44, 0.517, 0.565, 0.516, 0.121, 0.871]
    img_y_rate = [1 - v for v in img_y_rate]

    # Subject-wise
    sbj_y_true = [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    sbj_y_rate = [0.514, 0.826, 0.951, 0.183, 0.575, 0.307, 0.976, 0.496, 0.983, 0.684, 0.369, 0.964, 0.954, 0.953, 0.95, 0.679, 0.346, 0.764, 0.954, 0.42, 0.871, 0.895, 0.475, 0.393, 0.293, 0.506]
    sbj_y_rate = [1 - v for v in sbj_y_rate]

    print("Subject-wise:", len(sbj_y_true))
    print("Image-wise:", len(img_y_true))

    """ROC"""
    fpr, tpr, thresholds = roc_curve(img_y_true, img_y_rate, pos_label=pos_label)
    auc = roc_auc_score(img_y_true, img_y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label=f"Image-wise   (AUC={auc:.2})", linestyle='dashed')
    print(f"pos_label={pos_label}, AUC={auc:.3} @Image-wise")

    fpr, tpr, thresholds = roc_curve(sbj_y_true, sbj_y_rate, pos_label=pos_label)
    auc = roc_auc_score(sbj_y_true, sbj_y_rate)
    if pos_label == 0:
        auc = 1 - auc
    plt.plot(fpr, tpr, label=f"Subject-wise (AUC={auc:.2})", linestyle='solid')
    print(f"pos_label={pos_label}, AUC={auc:.3} @Subject-wise")

    # plt.title("ROC")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.axis('square')

    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)

    plt.legend()
    plt.tight_layout()

    plt.savefig('26-47_ROC.jpg', bbox_inches='tight')
    plt.savefig('26-47_ROC.pdf', bbox_inches='tight')
    plt.close()

    """Threshold vs. F-measure"""
    thresholds = list(np.arange(0, 1.0, 0.01))

    for label, (y_true, y_rate) in [
        ("Image-wise", (img_y_true, img_y_rate)),
        ("Subject-wise", (sbj_y_true, sbj_y_rate))
    ]:
        f1th = [
            f1_score(y_true=y_true, y_pred=[int(r > th) for r in y_rate])
            for th in thresholds
        ]
        print(f"{label:12} f1-max={max(f1th):.3} @th={thresholds[np.argmax(f1th)]}")

        plt.plot(thresholds, f1th, label=f"{label}")

    plt.ylabel("F-measure")
    plt.xlabel('Threshold')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid()
    plt.tick_params(direction='in')

    plt.legend()

    plt.savefig('26-47_f1th.jpg', bbox_inches='tight')
    plt.savefig('26-47_f1th.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
