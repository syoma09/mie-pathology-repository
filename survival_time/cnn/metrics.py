#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


class ConfusionMatrix(object):
    """
    +----------------+---------------------+
    |                |     Prediction      |
    |                | positive | negative |
    +--------+-------+----------+----------+
    | Ground | true  |    TP    |    FN    |
    | Truth  | false |    FP    |    TN    |
    +--------+-------+----------+----------+
    """

    def __init__(self, y_pred, y_true):
        # self._mat[y_pred, y_true]
        #   positive: 1
        #   negative: 0
        self._mat = np.zeros((2, 2), dtype=np.int64)

        if y_pred is not None and y_true is not None:
            # print(y_pred)
            # print(torch.max(y_pred, dim=1)[1])
            # print(y_true)
            # print(torch.max(y_true, dim=1))
            y_pred = torch.max(y_pred, dim=1)[1]    # argmax
            y_true = torch.max(y_true, dim=1)[1]    # argmax
            # print(y_pred)
            # print(y_true)
            for p, q in zip(y_pred, y_true):
                self._mat[p, q] += 1

    def __add__(self, obj):
        result = ConfusionMatrix(None, None)
        result._mat = self._mat + obj._mat

        return result

    def __str__(self):
        return "TP={:7} FN={:7}\nFP={:7} TN={:7}".format(
            self.tp, self.fn, self.fp, self.tn
        )

    @property
    def tp(self):
        return self._mat[1, 1]

    @property
    def fn(self):
        return self._mat[0, 1]

    @property
    def fp(self):
        return self._mat[1, 0]

    @property
    def tn(self):
        return self._mat[0, 0]

    def item(self):
        return self

    @property
    def tpr(self):
        """True-positive rate"""
        return self.tp / (self.tp + self.fn + 1e-8)

    @property
    def fpr(self):
        """False-positive rate"""
        return self.fp / (self.fp + self.tn + 1e-8)

    @property
    def tnr(self):
        """True-negative rate"""
        return self.tn / (self.tn + self.fp + 1e-8)

    @property
    def ppv(self):
        """positive predictive value"""
        return self.tp / (self.tp + self.fp + 1e-8)

    @property
    def npv(self):
        """negative predictive value"""
        return self.tn / (self.fn + self.tn + 1e-8)

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    @property
    def precision(self):
        return self.ppv

    @property
    def recall(self):
        return self.tpr

    @property
    def specificity(self):
        return self.tnr

    @property
    def f1(self):
        ppv = self.ppv
        tpr = self.tpr
        return 2 * ppv * tpr / (ppv + tpr + 1e-8)

    @property
    def f1inv(self):
        npv = self.npv
        tnr = self.tnr
        return 2 * npv * tnr / (npv + tnr + 1e-8)


# class Accuracy(ConfusionMatrix):
#     def __init__(self, y_pred, y_true):
#         super().__init__(y_pred, y_true)
# 
#     def item(self):
#         return super().accuracy()
# 
# 
# class F1(ConfusionMatrix):
#     def __init__(self, y_pred, y_true):
#         super().__init__(y_pred, y_true)
# 
#     def item(self):
#         return super().f1()
