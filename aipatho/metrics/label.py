#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeToLabelConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: float) -> int:
        raise NotImplementedError


class TimeToTime(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> float:
        return x


class TimeToFixed3(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x < 13:
            return 0
        elif x < 34:
            return 1
        else:   # elif x < 67:
            return 2


class TimeToFixed4(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x <= 11:
            return 0
        elif 11 < x <= 22:
            return 1
        elif 22 < x <= 33:
            return 2
        else:   # elif 33 < x <= 44:
            return 3


class TimeToFixed34(TimeToLabelConverter):
    def __init__(self):
        super().__init__()

    def __call__(self, x: float) -> int:
        if x < 11:
            return 0
        elif x < 22:
            return 1
        elif x < 33:
            return 2
        elif x < 44:
            return 3
        elif x < 44:
            return 4
        elif x < 36:
            return 5
        elif x < 42:
            return 6
        elif x < 48:
            return 7
        # ToDo: Where is 8 ~ 10
        elif x < 24:
            return 11
        elif x < 26:
            return 12
        elif x < 28:
            return 13
        elif x < 30:
            return 14
        elif x < 32:
            return 15
        elif x < 34:
            return 16
        elif x < 36:
            return 17
        elif x < 38:
            return 18
        elif x < 40:
            return 19
        elif x < 42:
            return 20
        elif x < 44:
            return 21
        elif x < 46:
            return 22
        elif x < 48:
            return 23
        elif x < 50:
            return 24
        elif x < 52:
            return 25
        elif x < 54:
            return 26
        elif x < 56:
            return 27
        elif x < 58:
            return 28
        elif x < 60:
            return 29
        elif x < 62:
            return 30
        elif x < 64:
            return 31
        elif x < 66:
            return 32
        else:   # elif x < 68:
            return 33


class LabelEncoder(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

class OneHot(LabelEncoder):
    def __init__(self, c: int):
        super().__init__()
        self._c = c     # Number of classes

    def __call__(self, x: int) -> torch.Tensor:
        return torch.eye(self._c)[x]


class GaussianSoft(nn.Module):
    def __init__(self, c: int, k: float = 1.0):
        """

        :param c:   Number of classes (e.g. c=4 for 4-class classification)
        :param k:   FIXME: Parameter...
        """
        super().__init__()

        self._c = c
        self._k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate: exp(-k * (x-i)^2)
        prob = torch.exp(
            -self._k * torch.pow(
                x.view(-1, 1) - torch.arange(0, self._c, 1),
                2.0
            )
        )
        # Normalize
        prob = prob / torch.sum(prob, dim=1, keepdim=True)

        return prob

if __name__ == '__main__':
    def main():
        label = GaussianSoftTorch(4).forward(torch.FloatTensor([0, 0, 1, 0, 2, 3]))
        print(label)

    main()
