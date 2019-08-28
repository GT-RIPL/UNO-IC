import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ptsemseg.models.utils import segnetDown2, segnetUp2

class Average(nn.Module):
    def __init__(self, n_classes):
        super(Average, self).__init__()

    def forward(self, mean, variance, mutual_info, entropy):

        return mean['rgb'] + mean['d']
        
class Multiply(nn.Module):
    def __init__(self, n_classes):
        super(Multiply, self).__init__()

    def forward(self, mean, variance, mutual_info, entropy):

        return mean['rgb'] * mean['d']

class NoisyOr(nn.Module):
    def __init__(self, n_classes):
        super(NoisyOr, self).__init__()

    def forward(self, mean, variance, mutual_info, entropy):

        return 1 - (1 - mean['rgb']) * (1 - mean['d'])

class GatedFusion(nn.Module):
    def __init__(self, n_classes):
        super(GatedFusion, self).__init__()

        self.conv = nn.Conv2d(
            2 * n_classes,
            n_classes,
            1,
            stride=1,
            padding=0,
            bias=True,
            dilation=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'].unsqueeze(1), entropy['rgb'].unsqueeze(1)
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'].unsqueeze(1), entropy['d'].unsqueeze(1)
        
        fusion = torch.cat([rgb, d], dim=1)

        G = self.conv(fusion)
        G = self.sigmoid(G)

        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = (P_rgb + P_d) / 2

        

        return P_fusion.log()


# 1.1
class ConditionalAttentionFusion(nn.Module):
    def __init__(self, n_channels):
        super(ConditionalAttentionFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * n_channels + 4,
                                            n_channels,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'].unsqueeze(1), entropy['rgb'].unsqueeze(1)
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'].unsqueeze(1), entropy['d'].unsqueeze(1)
        
        AB = torch.cat([rgb, d], dim=1)
        ABCD = torch.cat([rgb, d, rgb_mi, d_mi, d_entropy, rgb_entropy], dim=1)

        G = self.gate(ABCD)
        
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = (P_rgb + P_d) / 2

        return P_fusion.log()
# 1.2
class UncertaintyGatedFusion(nn.Module):
    def __init__(self, n_channels):
        super(UncertaintyGatedFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(6,
                                            n_channels,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'].unsqueeze(1), entropy['rgb'].unsqueeze(1)
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'].unsqueeze(1), entropy['d'].unsqueeze(1)
        
        CD = torch.cat([rgb_var, d_var, rgb_mi, d_mi, d_entropy, rgb_entropy], dim=1)

        G = self.gate(CD)
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = (P_rgb + P_d) / 2
        
        return P_fusion.log()


# 0.0
class TemperatureScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(TemperatureScaling,  self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, mean, variance, mutual_info, entropy):
    
        return mean / self.temperature

# 1.0
class LocalUncertaintyScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(LocalUncertaintyScaling, self).__init__()
        self.scale = nn.Conv2d(1,
                               1,
                               3,
                               stride=1,
                               padding=1,
                               bias=True)
                               
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        
        self.scale.weight = torch.nn.Parameter(torch.zeros((1,1,3,3)))
        if bias_init is not None:
            self.scale.bias = torch.nn.Parameter(bias_init)
        else:
            self.scale.bias = torch.nn.Parameter(torch.tensor([1.0]))
        

    def forward(self, mean, variance, mutual_info, entropy):
    
        x = torch.cat([entropy.unsqueeze(1)], dim=1)
        s = self.scale(x)
        out = mean / s
        out = self.norm(out)
        
        return out
        

class GlobalUncertaintyScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(GlobalUncertaintyScaling, self).__init__()
        self.scale = nn.Linear(1, 1)
        
        self.scale.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.scale.bias = torch.nn.Parameter(torch.tensor(0.0))
        
        self.norm = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, mean, variance, mutual_info, entropy):

        s = self.scale(entropy.mean().unsqueeze(0))
        out = mean / s
        out = self.norm(out)
        
        return out

class GlobalLocalUncertaintyScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(GlobalLocalUncertaintyScaling, self).__init__()
        
        self.scale_local = nn.Conv2d(1,
                               1,
                               3,
                               stride=1,
                               padding=1,
                               bias=False)
                               
        self.scale_local.weight = torch.nn.Parameter(torch.zeros((1,1,3,3)))
        # self.scale_local.bias = torch.nn.Parameter(torch.tensor([1.0]))
        
        
        self.scale_global = nn.Linear(1, 1)
        self.scale_global.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.scale_global.bias = torch.nn.Parameter(torch.tensor(0.0))
        
        self.norm = nn.Sequential(nn.Softmax(dim=1))

    def forward(self, mean, variance, mutual_info, entropy):
        s_local = self.scale_local(entropy.unsqueeze(1))
        s_global = self.scale_global(entropy.mean().unsqueeze(0))
        out = mean / (s_local + s_global)
        out = self.norm(out)
        
        return out
        
class UncertaintyScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(UncertaintyScaling, self).__init__()
        self.d1 = segnetDown2(1, 64)
        self.d2 = segnetDown2(64, 128)
        self.u2 = segnetUp2(128, 64)
        self.u1 = segnetUp2(64, 1)
                               
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        

    def forward(self, mean, variance, mutual_info, entropy):
    
        x = torch.cat([entropy.unsqueeze(1)], dim=1)
        tdown1, tindices_1, tunpool_shape1 = self.d1(x)
        tdown2, tindices_2, tunpool_shape2 = self.d2(tdown1)

        tup2 = self.u2(tdown2, tindices_2, tunpool_shape2)
        tup1 = self.u1(tup2, tindices_1, tunpool_shape1)  # [batch,1,512,512]

        avg_temp = tup1.mean((2, 3)).unsqueeze(-1).unsqueeze(-1)  # (batch,1,1,1)
        
        out = mean * tup1
        out = self.norm(out)
        
        return out
        

    