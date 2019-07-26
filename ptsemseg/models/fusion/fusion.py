import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, n_classes, compression_rate=1):
        super(GatedFusion, self).__init__()

        self.conv = nn.Conv2d(
            2 * n_classes,
            n_classes,
            1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d):

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
    def __init__(self, n_channels, compression_rate=1):
        super(ConditionalAttentionFusion, self).__init__()
        self.bottleneck = nn.Conv2d(
            2 * n_channels + 2,
            n_channels // compression_rate,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        
        
        self.gate = nn.Conv2d(
            n_channels // compression_rate,
            2 * n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        
        conv_mod = nn.Conv2d(
            2 * n_channels,
            n_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1
        )
        
        self.fuser = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_channels)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d, rgb_var, d_var):

        AB = torch.cat([rgb, d], dim=1)
        ABCD = torch.cat([rgb, d, rgb_var, d_var], dim=1)
        
        G = self.bottleneck(ABCD)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G
        
        fused = self.fuser(AB)

        return fused

# 1.2
class PreweightedGatedFusion(nn.Module):
    def __init__(self, n_channels, compression_rate=1):
        super(PreweightedGatedFusion, self).__init__()
        self.bottleneck = nn.Conv2d(
            2 * n_channels,
            n_channels // compression_rate,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )

        self.gate = nn.Conv2d(
            n_channels // compression_rate,
            2 * n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )

        conv_mod = nn.Conv2d(
            2 * n_channels,
            n_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1
        )

        self.fuser = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_channels)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d, rgb_var, d_var):

        rgb_var = 1 / (rgb_var + 1e-5)
        d_var = 1 / (d_var + 1e-5)

        for n in range(rgb.shape[1]):
            rgb[:, n, :, :] = rgb[:, n, :, :] * rgb_var
            d[:, n, :, :] = d[:, n, :, :] * d_var

        AB = torch.cat([rgb, d], dim=1)

        G = self.bottleneck(AB)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G

        fused = self.fuser(fusion)

        return fused
# 1.3
class UncertaintyGatedFusion(nn.Module):
    def __init__(self, n_channels, compression_rate=1):
        super(UncertaintyGatedFusion, self).__init__()
        self.bottleneck = nn.Conv2d(
            2 * n_channels,
            n_channels // compression_rate,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        
        
        self.gate = nn.Conv2d(
            n_channels // compression_rate,
            2 * n_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            dilation=1
        )
        
        conv_mod = nn.Conv2d(
            2 * n_channels,
            n_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
            dilation=1
        )
        
        self.fuser = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_channels)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, d, rgb_var, d_var):


        AB = torch.cat([rgb, d], dim=1)
        CD = torch.cat([rgb_var, d_var], dim=1)
       
        G = self.bottleneck(CD)
        G = F.relu(G)
        G = self.gate(G)
        G = self.sigmoid(G)

        AB = AB * G
        
        fused = self.fuser(fusion)

        return fused
        