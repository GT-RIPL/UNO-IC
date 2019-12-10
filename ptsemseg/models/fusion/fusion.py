import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ptsemseg.models.utils import segnetDown2, segnetUp2

class Average(nn.Module):
    def __init__(self, n_classes):
        super(Average, self).__init__()

    def forward(self, mean, variance, entropy, mutual_info):

        return mean['rgb'] + mean['d']
        
class Multiply(nn.Module):
    def __init__(self, n_classes):
        super(Multiply, self).__init__()

    def forward(self, mean, variance, entropy, mutual_info):

        return mean['rgb'] * mean['d']

class NoisyOr(nn.Module):
    def __init__(self, n_classes):
        super(NoisyOr, self).__init__()

    def forward(self, mean, variance, entropy, mutual_info):
        x = (1 - (1 - mean['rgb']) * (1 - mean['d']))
        x = x.masked_fill(x < 1e-9, 1e-9)
        return x.log()

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

    def forward(self, mean, variance, entropy, mutual_info):
    
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
    def __init__(self, n_classes):
        super(ConditionalAttentionFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * n_classes + 4,
                                            n_classes,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, entropy, mutual_info):
    
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
    def __init__(self, n_classes):
        super(UncertaintyGatedFusion, self).__init__()
        self.gate = nn.Sequential(nn.Conv2d(6,
                                            n_classes,
                                            3,
                                            stride=1,
                                            padding=1,
                                            bias=True,
                                            dilation=1),
                                  nn.Sigmoid())

    def forward(self, mean, variance, entropy, mutual_info):
    
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


# 2.2
class FullyUncertaintyGatedFusion(nn.Module):
    def __init__(self, n_classes):
        super(FullyUncertaintyGatedFusion, self).__init__()
        self.d1 = segnetDown2(2, 64)
        self.d2 = segnetDown2(64, 128)
        self.u2 = segnetUp2(128, 64)
        self.u1 = segnetUp2(64, n_classes)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, mean, variance, entropy, mutual_info):
    
        rgb, rgb_var, rgb_mi, rgb_entropy = mean['rgb'], variance['rgb'], mutual_info['rgb'].unsqueeze(1), entropy['rgb'].unsqueeze(1)
        d, d_var, d_mi, d_entropy = mean['d'], variance['d'], mutual_info['d'].unsqueeze(1), entropy['d'].unsqueeze(1)
        
        uncertainty = torch.cat([d_entropy, rgb_entropy], dim=1)
        
        down1, indices_1, unpool_shape1 = self.d1(uncertainty)
        down2, indices_2, unpool_shape2 = self.d2(down1)
        up2 = self.u2(down2, indices_2, unpool_shape2)
        up1 = self.u1(up2, indices_1, unpool_shape1)
        G = self.sigmoid(up1)
        
        G_rgb = G
        G_d = torch.ones(G_rgb.shape, dtype=torch.float, device=G_rgb.device) - G_rgb

        # take weighted average of probabilities
        P_rgb = rgb * G_rgb
        P_d = d * G_d

        P_fusion = (P_rgb + P_d) / 2


        return P_fusion.log()


# 0.0
class TemperatureScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(TemperatureScaling,  self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, mean, variance, entropy, mutual_info):
    
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
        

    def forward(self, mean, variance, entropy, mutual_info):
    
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

    def forward(self, mean, variance, entropy, mutual_info):

        s = self.scale(entropy.mean().unsqueeze(0))
        out = mean / s
        out = self.norm(out)
        out = out.masked_fill(out < 1e-9, 1e-9)
        
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

    def forward(self, mean, variance, entropy, mutual_info):
        s_local = self.scale_local(entropy.unsqueeze(1))
        s_global = self.scale_global(entropy.mean().unsqueeze(0))
        out = mean / (s_local + s_global)
        out = self.norm(out)
        out = out.masked_fill(out < 1e-9, 1e-9)
        
        return out
        
class UncertaintyScaling(nn.Module):
    def __init__(self, n_classes=11, bias_init=None):
        super(UncertaintyScaling, self).__init__()
        self.d1 = segnetDown2(1, 64)
        self.d2 = segnetDown2(64, 128)
        self.u2 = segnetUp2(128, 64)
        self.u1 = segnetUp2(64, 1)
                               
        self.norm = nn.Sequential(nn.Softmax(dim=1))
        

    def forward(self, mean, variance, entropy, mutual_info):
    
        x = torch.cat([entropy.unsqueeze(1)], dim=1)
        tdown1, tindices_1, tunpool_shape1 = self.d1(x)
        tdown2, tindices_2, tunpool_shape2 = self.d2(tdown1)

        tup2 = self.u2(tdown2, tindices_2, tunpool_shape2)
        tup1 = self.u1(tup2, tindices_1, tunpool_shape1)  # [batch,1,512,512]
        
        out = mean * tup1
        out = self.norm(out)
        
        out = out.masked_fill(out < 1e-9, 1e-9)
        
        return out


class GlobalScaling(nn.Module):
    def __init__(self,train_stats):
        super(GlobalScaling, self).__init__()
        self.MI_MEAN_rgb = train_stats['MI_MEAN_rgb']# 0.01635716 #0.010646423
        self.MI_STD_rgb = train_stats['MI_STD_rgb']#0.00688986 #0.004948631
        self.PreEn_MEAN_rgb = train_stats['PreEn_MEAN_rgb']#0.16158119 #0.098073044 
        self.PreEn_STD_rgb = train_stats['PreEn_STD_rgb']#0.03454985 #0.031052864
        self.SoftEn_MEAN_rgb = train_stats['SoftEn_MEAN_rgb']#0.10871122 #0.072535
        self.SoftEn_STD_rgb = train_stats['SoftEn_STD_rgb']#0.02284632 #0.024109
        # self.Temp_MEAN_rgb = 0.80496331 #0.789093009
        # self.Temp_STD_rgb = 0.02468624 #0.016560384
        self.MI_MEAN_d = train_stats['MI_MEAN_d']#0.02271291 #0.015116896
        self.MI_STD_d = train_stats['MI_STD_d']#0.01019019 #0.00831911
        self.PreEn_MEAN_d = train_stats['PreEn_MEAN_d']#0.20687151 #0.119714723 
        self.PreEn_STD_d = train_stats['PreEn_STD_d']#0.04621102 #0.034841593
        self.SoftEn_MEAN_d = train_stats['SoftEn_MEAN_d']# 0.13952574 #0.085331
        self.SoftEn_STD_d = train_stats['SoftEn_STD_d']#0.03168533 #0.022258
        # self.Temp_MEAN_d =  0.84158579 #0.80703
        # self.Temp_STD_d = 0.01058592 #0.035887
        self.SoftEn_MEAN_rgbd = train_stats['SoftEn_MEAN_rgbd']#0.05217332 #0.085331
        self.SoftEn_STD_rgbd = train_stats['SoftEn_STD_rgbd']#0.01356846 #0.022258
        # self.Temp_MEAN_rgb_rgbd = train_stats['Temp_MEAN_rgb_rgbd']#0.80496331 #0.789093009
        # self.Temp_STD_rgb_rgbd = train_stats['Temp_STD_rgb_rgbd']#0.02468624 #0.016560384
        # self.Temp_MEAN_rgbd =  0.84158579 #0.80703
        # self.Temp_STD_rgbd = 0.01058592 #0.035887
        
    def forward(self, entropy, mutual_info, modality,mode='mixed'):

        if modality == 'rgb': 
            MI_MEAN = self.MI_MEAN_rgb# 0.01635716 #0.010646423
            MI_STD = self.MI_STD_rgb#0.00688986 #0.004948631

            PreEn_MEAN = self.PreEn_MEAN_rgb #0.16158119 #0.098073044 
            PreEn_STD = self.PreEn_STD_rgb#0.03454985 #0.031052864

            SoftEn_MEAN = self.SoftEn_MEAN_rgb#0.10871122 #0.072535
            SoftEn_STD = self.SoftEn_STD_rgb#0.02284632 #0.024109
            # self.Temp_MEAN = 0.80496331 #0.789093009
            # self.Temp_STD = 0.02468624 #0.016560384
        elif modality == 'd':
            MI_MEAN = self.MI_MEAN_d#0.02271291 #0.015116896
            MI_STD = self.MI_STD_d#0.01019019 #0.00831911

            PreEn_MEAN = self.PreEn_MEAN_d#0.20687151 #0.119714723 
            PreEn_STD = self.PreEn_STD_d#0.04621102 #0.034841593

            SoftEn_MEAN = self.SoftEn_MEAN_d# 0.13952574 #0.085331
            SoftEn_STD = self.SoftEn_STD_d#0.03168533 #0.022258

            # self.Temp_MEAN =  0.84158579 #0.80703
            # self.Temp_STD = 0.01058592 #0.035887

        elif modality == 'rgbd':
            SoftEn_MEAN = self.SoftEn_MEAN_rgbd#0.05217332 #0.085331
            SoftEn_STD = self.SoftEn_STD_rgbd#0.01356846 #0.022258
            # self.Temp_MEAN_rgb = self.Temp_MEAN_rgb#0.80496331 #0.789093009
            # self.Temp_STD_rgb = self.Temp_STD_rgb#0.02468624 #0.016560384

        if mode == "MI":
            STD_MEAN = torch.max(torch.zeros_like(mutual_info.mean((1,2))),mutual_info.mean((1,2)) - MI_MEAN - MI_STD)+MI_MEAN
            DR = MI_MEAN/STD_MEAN.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        elif mode == 'PreEn':
            STD_MEAN = torch.max(torch.zeros_like(entropy.mean((1,2))),entropy.mean((1,2)) - PreEn_MEAN - PreEn_STD)+PreEn_MEAN
            DR = PreEn_MEAN/STD_MEAN.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        elif mode == "SoftEn":
            STD_MEAN = torch.max(torch.zeros_like(entropy.mean((1,2))),entropy.mean((1,2)) - SoftEn_MEAN - SoftEn_STD)+SoftEn_MEAN
            DR = SoftEn_MEAN/STD_MEAN.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        elif mode == "AveTemp":
            STD_MEAN = torch.min(torch.zeros_like(temp1.mean((1,2))), temp1.mean((1,2))+ Temp_STD - Temp_MEAN ) + Temp_MEAN
            DR = STD_MEAN.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/Temp_MEAN
        else:
            #STD_MEAN_Temp = torch.min(torch.zeros_like(temp1.mean((1,2))), temp1.mean((1,2))+ self.Temp_STD - self.Temp_MEAN ) + self.Temp_MEAN
            #DR_Temp = STD_MEAN_Temp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/self.Temp_MEAN

            # STD_MEAN_En = torch.max(torch.zeros_like(entropy.mean((1,2))),entropy.mean((1,2)) - self.PreEn_MEAN - self.PreEn_STD)+self.PreEn_MEAN
            # DR_En = self.PreEn_MEAN/STD_MEAN_En.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            # STD_MEAN_MI = torch.max(torch.zeros_like(mutual_info.mean((1,2))),mutual_info.mean((1,2)) - self.MI_MEAN - self.MI_STD)+self.MI_MEAN
            # DR_MI = self.MI_MEAN/STD_MEAN_MI.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # DR = torch.min(DR_Temp,torch.min(DR_MI,DR_En))
            #if self.modality != 'rgbd':
            # STD_MEAN_Temp = torch.min(torch.zeros_like(temp1.mean((1,2))), temp1.mean((1,2))+ Temp_STD - Temp_MEAN ) + Temp_MEAN
            # DR_Temp = STD_MEAN_Temp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)/Temp_MEAN

            STD_MEAN_En= torch.max(torch.zeros_like(entropy.mean((1,2))),entropy.mean((1,2)) - SoftEn_MEAN - SoftEn_STD)+SoftEn_MEAN
            DR_En = SoftEn_MEAN/STD_MEAN_En.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            DR = torch.min(DR_En,DR_Temp)
            
        return DR
        

def fusion(fusion_type,mean):
    if fusion_type== "None":
        if 'rgbd' in mean:
            outputs = torch.nn.Softmax(dim=1)(mean['rgbd'])
        else:
            outputs = torch.nn.Softmax(dim=1)(mean['rgb'])
    elif fusion_type == "SoftmaxMultiply":
        outputs = torch.nn.Softmax(dim=1)(mean["rgb"]) * torch.nn.Softmax(dim=1)(mean["d"]) 
        if 'rgbd' in mean:
            outputs = outputs * torch.nn.Softmax(dim=1)(mean["rgbd"])
    elif fusion_type == "SoftmaxAverage":
        outputs = torch.nn.Softmax(dim=1)(mean["rgb"]) + torch.nn.Softmax(dim=1)(mean["d"])
        if 'rgbd' in mean:
            outputs = outputs + torch.nn.Softmax(dim=1)(mean["rgbd"])
    elif fusion_type == "Noisy-Or":
        if 'rgbd' in mean:
            outputs = 1 - (1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) * (1 - torch.nn.Softmax(dim=1)(mean["rgbd"])) #[batch,11,512,512,1]
        else:
            outputs = 1 - (1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) #[batch,11,512,512,1]
    elif fusion_type == "Stacked-Noisy-Or":
        soft = torch.nn.Softmax(dim=1)(mean["rgb"]) * torch.nn.Softmax(dim=1)(mean["d"]) * torch.nn.Softmax(dim=1)(mean["rgbd"])#[batch,11,512,512,1]
        soft = soft/soft.sum(1).unsqueeze(1)
        if 'rgbd' in mean:
            outputs = 1 - (1-soft)*(1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) * (1 - torch.nn.Softmax(dim=1)(mean["rgbd"])) #[batch,11,512,512,1]
        else:
            outputs = 1 - (1-soft)*(1 - torch.nn.Softmax(dim=1)(mean["rgb"])) * (1 - torch.nn.Softmax(dim=1)(mean["d"])) #[batch,11,512,512,1]
    elif cfg["fusion"] == "BayesianGMM":      
        pass
    else:
        print("Fusion Type Not Supported")
    return outputs