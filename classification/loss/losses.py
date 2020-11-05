import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight #8142

    def forward(self, input, target):
        # targe: (batch,)
        CE = F.cross_entropy(input, target, reduction='none')
        p = torch.exp(-CE)
        if self.weight is not None:
            loss = (1 - p) ** self.gamma * self.weight[target] * CE
        else: 
            loss = (1 - p) ** self.gamma * CE
        return loss.mean()


        