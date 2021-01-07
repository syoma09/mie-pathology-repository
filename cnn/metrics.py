#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


class ConfusionMatrix(object):
    def __init__(self, y_pred, y_true):
        """
        +----------------+---------------------+
        |                |     Prediction      |
        |                | positive | negative |
        +--------+-------+----------+----------+
        | Ground | true  |    TP    |    FN    |
        | Truth  | false |    FP    |    TN    |
        +--------+-------+----------+----------+
        """
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
            # y_true = torch.max(y_true, dim=1)[1]    # argmax
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

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

    def precision(self):
        return self.tp / (self.tp + self.fp + 1e-8)

    def recall(self):
        return self.tp / (self.tp + self.fn + 1e-8)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-8)


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
