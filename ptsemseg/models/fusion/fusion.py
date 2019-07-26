import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, n_classes):
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
    def __init__(self, n_channels):
        super(ConditionalAttentionFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * n_channels + 2,
                                    n_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)),
                                    nn.ReLU(),
                                    nn.Sigmoid())
                     
        self.fuser = nn.Sequential(nn.Conv2d(n_channels,
                                    n_channels,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)))

    def forward(self, rgb, d, rgb_var, d_var):

        AB = torch.cat([rgb, d], dim=1)
        ABCD = torch.cat([rgb, d, rgb_var, d_var], dim=1)
        
        G = self.gate(ABCD)
        # fused = self.fuser(G)

        # return fused
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion

# 1.2
class PreweightedGatedFusion(nn.Module):
    def __init__(self, n_channels):
        super(PreweightedGatedFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * n_channels,
                                    n_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)),
                                    nn.ReLU(),
                                    nn.Sigmoid())
                     
        self.fuser = nn.Sequential(nn.Conv2d(n_channels,
                                    n_channels,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)))

    def forward(self, rgb, d, rgb_var, d_var):

        rgb_var = 1 / (rgb_var + 1e-5)
        d_var = 1 / (d_var + 1e-5)

        for n in range(rgb.shape[1]):
            rgb[:, n, :, :] = rgb[:, n, :, :] * rgb_var
            d[:, n, :, :] = d[:, n, :, :] * d_var

        AB = torch.cat([rgb, d], dim=1)
        
        G = self.gate(ABCD)
        # fused = self.fuser(G)

        # return fused
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion
# 1.3
class UncertaintyGatedFusion(nn.Module):
    def __init__(self, n_channels):
        super(UncertaintyGatedFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2,
                                    n_channels,
                                    3,
                                    stride=1,
                                    padding=1,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)),
                                    nn.ReLU(),
                                    nn.Sigmoid())
                     
        self.fuser = nn.Sequential(nn.Conv2d(n_channels,
                                    n_channels,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)))

    def forward(self, rgb, d, rgb_var, d_var):


        AB = torch.cat([rgb, d], dim=1)
        CD = torch.cat([rgb_var, d_var], dim=1)
       
        G = self.gate(CD)
        # fused = self.fuser(G)
        
        # return fused
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion

#2.1
class ConditionalAttentionFusionv2(nn.Module):
    def __init__(self, n_channels):
        super(ConditionalAttentionFusionv2, self).__init__()
        self.probability_fusion = []
        self.uncertainty_fusion = []
        self.total_fusion = []
        for n in range(n_channels):
        
            self.uncertainty_fusion.append(nn.Conv2d(2,
                                        1,
                                        3,
                                        stride=1,
                                        padding=1,
                                        bias=False,
                                        dilation=1).cuda())
                                        
            self.probability_fusion.append(nn.Conv2d(2,
                                        1,
                                        1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                        dilation=1).cuda())

            self.total_fusion.append(nn.Conv2d(2,
                                                1,
                                                1,
                                                stride=1,
                                                padding=0,
                                                bias=False,
                                                dilation=1).cuda())
                     
        self.fuser = nn.Sequential(nn.Conv2d(n_channels,
                                    n_channels,
                                    1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    dilation=1),
                                    nn.BatchNorm2d(int(n_channels)))
                                    
        self.n_channels = n_channels

    def forward(self, rgb, d, rgb_var, d_var):

        AB = torch.cat([rgb, d], dim=1)
        CD = torch.cat([rgb_var, d_var], dim=1)
        G = torch.zeros(rgb.shape, device=AB.device)
        
        for n in range(self.n_channels):
        
            A = rgb[:, n, :, :].view(-1, 1, rgb.shape[2], rgb.shape[3])
            B = d[:, n, :, :].view(-1, 1, d.shape[2], d.shape[3])
            
            _AB = self.probability_fusion[n](torch.cat([A, B], dim=1))
            _CD = self.uncertainty_fusion[n](CD)
            
            G[:, n, :, :] = torch.squeeze(self.total_fusion[n](torch.cat([_AB, _CD], dim=1)))
        
        # fused = self.fuser(G)

        # return fused
        G_rgb = G
        G_d = torch.ones(G.shape, dtype=torch.float, device=G.device) - G

        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = P_rgb + P_d

        return P_fusion


# 2.2
class PreweightedUncertaintyFusionv2(nn.Module):
    def __init__(self, n_channels):
        super(PreweightedUncertaintyFusionv2, self).__init__()

        self.rgb_uncertainty = nn.Sequential(nn.Conv2d(2,
                                             1,
                                             5,
                                             stride=1,
                                             padding=2,
                                             bias=True,
                                             dilation=1))
        self.d_uncertainty = nn.Sequential(nn.Conv2d(2,
                                             1,
                                             5,
                                             stride=1,
                                             padding=2,
                                             bias=True,
                                             dilation=1))

        self.n_channels = n_channels

    def forward(self, rgb, d, rgb_var, d_var):

        CD = torch.cat([rgb_var, d_var], dim=1)

        # compute uncertainty weighted estimates
        rgb_weight = self.rgb_uncertainty(CD)
        d_weight = self.d_uncertainty(CD)
 
        # duplicate to each of the classes
        rgb_weight = rgb_weight.expand(rgb.shape[0], self.n_channels, rgb.shape[2], rgb.shape[3])
        d_weight = d_weight.expand(d.shape[0], self.n_channels, d.shape[2], d.shape[3])

        # weight by uncertainty fused
        rgb = rgb * rgb_weight
        d = d * d_weight

        return rgb + d