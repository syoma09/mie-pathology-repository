#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import display

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def calc_metric(y_true, y_pred, swap=False):
  # Swap 0/1
  if swap:
    y_true = np.array([1, 0])[y_true]
    y_pred = np.array([1, 0])[y_pred]

  print('y_true:', y_true)
  print('y_pred:', y_pred)

  cmat = confusion_matrix(y_true, y_pred)
  print(f'F-measure: {f1_score(y_true, y_pred):.3}')
  print(f'Accuracy : {accuracy_score(y_true, y_pred):.3}')
  print(f'Recall   : {recall_score(y_true, y_pred):.3}')
  print(f'Precision: {precision_score(y_true, y_pred):.3}')

  print(
    pd.DataFrame(
      cmat,
      columns=[
        ['Prediction'] * 2,
        ['0', '1']
      ],
      index=[
        ['Truth'] * 2,
        ['0', '1']
      ]
    )
  )


def main():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0])
    print("-- Subject-wise (F1, SGD, Death detection) -------------------------")
    calc_metric(
        y_true,
        y_pred=np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0]),
        swap=False
    )
    print("-- Subject-wise (F1, Adam, Death detection) -------------------------")
    calc_metric(
        y_true,
        y_pred=np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]),
        swap=False
    )
    print("-- Subject-wise (F1inv, SGD, Death detection) -----------------------")
    calc_metric(
        y_true,
        y_pred=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
        swap=False
    )
    print("-- Subject-wise (F1inv, Adam, Death detection) ----------------------")
    calc_metric(
        y_true,
        y_pred=np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]),
        swap=False
    )



if __name__ == '__main__':
    main()