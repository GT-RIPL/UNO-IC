import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ScaledAverage(nn.Module):
    def __init__(self, n_classes):
        super(ScaledAverage, self).__init__()
        self.rgb_scaling = torch.nn.Parameter(torch.ones(1))
        self.d_scaling = torch.nn.Parameter(torch.ones(1))

    def forward(self, mean, variance, mutual_info, entropy):

        return mean['rgb'] * self.rgb_scaling + mean['d'] * self.d_scaling

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
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.rgb_temperature = nn.Parameter(torch.ones(1))
        self.d_temperature = nn.Parameter(torch.ones(1))

    def forward(self, mean, variance, mutual_info, entropy):
    
        return mean['rgb'] / self.rgb_temperature, mean['d'] / self.d_temperature 

# 1.0
class UncertaintyScaling(nn.Module):
    def __init__(self):
        super(UncertaintyScaling, self).__init__()
        self.rgb_scale = nn.Conv2d(2,
                                   1,
                                   3,
                                   stride=1,
                                   padding=1,
                                   bias=True,
                                   dilation=1)
        self.d_scale = nn.Conv2d(2,
                                 1,
                                 3,
                                 stride=1,
                                 padding=1,
                                 bias=True,
                                 dilation=1)

    def forward(self, mean, variance, mutual_info, entropy):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'], entropy['rgb']
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'], entropy['d']
        
        rgb = rgb / self.rgb_scale(torch.cat([rgb_var, rgb_entropy], dim=1))
        d = d / self.d_scale(torch.cat([d_var, d_entropy], dim=1))
        
        return rgb, d
        