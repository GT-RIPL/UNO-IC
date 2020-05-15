import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            # input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            input = F.upsample(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=self.weight, size_average=self.size_average, ignore_index=250
        )
        return loss


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average = True, gamma=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average


    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            # input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            input = F.upsample(input, size=(ht, wt), mode="bilinear", align_corners=True)
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight,size_average=self.size_average), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, size_average = True,max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list[m_list==np.inf] = 0.0
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            # input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
            input = F.upsample(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)

        index = torch.zeros_like(input, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = input - batch_m
        output = torch.where(index, x_m, input)

        return F.cross_entropy(self.s*output, target, weight=self.weight,size_average=self.size_average)
