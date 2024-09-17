#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


class MeanVarianceLoss(nn.Module):
    """
    Original implementation at:
        https://github.com/Herosan163/AgeEstimation/blob/master/mean_variance_loss.py
    """
    def __init__(self, lambda1: float, lambda2: float, start_age, end_age):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.start_age = start_age
        self.end_age = end_age


    # ToDo: Check implementation and usage
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, device: str):
        # target = target.type(torch.FloatTensor)

        # m = nn.Softmax(dim=1)
        # p = m(y_pred)
        p = F.softmax(y_pred, dim=1)

        # Mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).to(device)
        # テンソルのサイズを確認
        """print(f"p size: {p.size()}")
        print(f"a size: {a.size()}")"""
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - y_true)**2
        mean_loss = mse.mean() / 2.0

        # Variance loss
        # print(a[None, :])
        # print(mean[:, None])
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda1 * mean_loss, self.lambda2 * variance_loss
