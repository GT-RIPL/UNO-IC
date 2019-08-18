import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'], entropy['rgb']
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'], entropy['d']
        
        fusion = torch.cat([rgb, d], dim=1)

        G = self.conv(fusion)
        G = self.sigmoid(G)

        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion


# 1.1
class ConditionalAttentionFusion(nn.Module):
    def __init__(self, n_channels):
        super(ConditionalAttentionFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * n_channels + 2,
                                            n_channels,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'], entropy['rgb']
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'], entropy['d']
        
        AB = torch.cat([rgb, d], dim=1)
        ABCD = torch.cat([rgb, d, rgb_var, d_var], dim=1)

        G = self.gate(ABCD)
        
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion

# 1.2
class UncertaintyGatedFusion(nn.Module):
    def __init__(self, n_channels):
        super(UncertaintyGatedFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2,
                                            n_channels,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'], entropy['rgb']
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'], entropy['d']
        
        CD = torch.cat([rgb_var, d_var], dim=1)

        G = self.gate(CD)
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion


# 0.0
class TemperatureScaling(nn.Module):
    def __init__(self, rgb_init=None, d_init=None):
        super(TemperatureScaling, self).__init__()
        self.rgb_temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.d_temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, mean, variance, mutual_info, entropy):
    
        return mean['rgb'] / self.rgb_temperature, mean['d'] / self.d_temperature 

# 1.0
class UncertaintyScaling(nn.Module):
    def __init__(self, rgb_init=None, d_init=None):
        super(UncertaintyScaling, self).__init__()
        self.rgb_scale = nn.Conv2d(1,
                                   1,
                                   3,
                                   stride=1,
                                   padding=1,
                                   bias=True,
                                   dilation=1)
        self.d_scale = nn.Conv2d(1,
                                 1,
                                 3,
                                 stride=1,
                                 padding=1,
                                 bias=True,
                                 dilation=1)
        
        
        self.rgb_scale.weight = torch.nn.Parameter(torch.zeros((1,1,3,3)))
        
        self.d_scale.weight = torch.nn.Parameter(torch.zeros((1,1,3,3)))
          
        if rgb_init is not None:
            self.rgb_scale.bias = torch.nn.Parameter(rgb_init)
        else:                       
            self.rgb_scale.bias = torch.nn.Parameter(torch.ones(1))
        if d_init is not None:
            self.d_scale.bias = torch.nn.Parameter(d_init)
        else:
            self.d_scale.bias = torch.nn.Parameter(torch.ones(1))
        
        

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'], entropy['rgb']
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'], entropy['d']
        
        # rgb_s = self.rgb_scale(torch.cat([rgb_var, rgb_entropy.unsqueeze(1)], dim=1))
        # d_s = self.d_scale(torch.cat([d_var, d_entropy.unsqueeze(1)], dim=1))
        rgb_s = self.rgb_scale(rgb_var)
        d_s = self.d_scale(d_var)
        print("rgb weight: {}".format(self.rgb_scale.weight.mean()))
        print("d weight: {}".format(self.d_scale.weight.mean()))
        print("rgb bias: {}".format(self.rgb_scale.bias))
        print("d bias: {}".format(self.d_scale.bias))
        print("---------------------")
        rgb = rgb / rgb_s
        d = d / d_s
        
        return rgb, d
        