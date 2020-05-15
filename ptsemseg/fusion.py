import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ptsemseg.models.utils import segnetDown2, segnetUp2
from scipy.special import logsumexp

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
        
def compute_log_normal(cfg,inputs,det_cov,inv_cov,mean,cl):
        # inputs [batch,num_class,760,1280]
        inputs = inputs.transpose(2,1).transpose(3,2).reshape(-1,14)
        temp = inputs - mean[cl].T # (batch,10)

        cnst = -0.5*(14*np.log(2*3.14) + torch.log(det_cov[cl])) #float
        cnst2 = -inputs.sum(1) +  torch.logsumexp(inputs, 1, keepdim=False)*14#(torch.log(temp1)+temp1_max.squeeze())* 14# batch
        temp =  cnst-1/2 * (torch.mm(temp,inv_cov[cl]) * temp).sum(1)
        temp = temp.reshape(-1,cfg['data']['img_cols'],cfg['data']['img_rows'])
        # import ipdb;ipdb.set_trace()
        return temp

# def compute_log_normal_fixed(cfg,inputs,det_cov,inv_cov,mean,cl):
#         # inputs [batch,num_class,760,1280]
        
#         inputs = inputs.transpose(2,1).transpose(3,2).reshape(-1,14)
#         temp = inputs - mean[cl].T # (batch,10)
#         # temp1_max =  torch.max(inputs,dim=1,keepdim=True)[0]
#         # temp1 = torch.exp(inputs - temp1_max).sum(1)  
#         cnst = -0.5*(14*np.log(2*3.14) + torch.log(det_cov)) #float
#         cnst2 = -inputs.sum(1) +  torch.logsumexp(inputs, 1, keepdim=False)#(torch.log(temp1)+temp1_max.squeeze())* 14# batch
        
#         temp =  cnst -1/2 * (torch.mm(temp,inv_cov) * temp).sum(1)
#         temp = temp.reshape(-1,cfg['data']['img_cols'],cfg['data']['img_rows'])
#         return temp



def uncertainty(mean,cfg,**kargs):
    outputs = {}
    for m in cfg["models"].keys():
        if not cfg['uncertainty']:
            outputs[m] = torch.nn.Softmax(dim=1)(mean[m])
        else:
            log_liklihood = torch.zeros((mean[m].shape[0],mean[m].shape[1],cfg['data']['img_cols'],cfg['data']['img_rows']),device=kargs['device'])
            mean_temp = np.delete(mean[m].cpu(),[13,14],1).to("cuda")
            for cl in range(16):
                if cl != 13 and cl != 14:
                    log_liklihood[:,cl,:,:] = compute_log_normal(cfg,mean_temp,kargs['det_cov'][m],kargs['inv_cov'][m],kargs['mean_stats'][m],cl)#[batch,11,512,512,1]              

            posterior = torch.nn.Softmax(dim=1)(mean[m])
            likelihood = torch.exp(log_liklihood+kargs['log_prior'])#*posterior #*torch.exp()
            # outputs[m] = joint/joint.sum(1).unsqueeze(1)
            marginal = likelihood.sum(1).unsqueeze(1)
            outputs[m] = posterior*marginal #
            # outputs[m][:,13:15,:,:] = 0.0
            # import ipdb;ipdb.set_trace()
            # outputs[m] = outputs[m]/outputs[m].sum(1).unsqueeze(1)
    return outputs

def fusion(mean,cfg,**kargs):
    if cfg["fusion"] == "None":
        outputs = mean[list(cfg["models"].keys())[0]]
    elif cfg["fusion"] == "SoftmaxMultiply":
        outputs = mean["rgb"] * mean["d"]
        if 'rgbd' in mean:
            outputs = outputs * mean["rgbd"]
    elif cfg["fusion"] == "SoftmaxAverage":
        outputs = mean["rgb"] + mean["d"]
        if 'rgbd' in mean:
            outputs = outputs + mean["rgbd"]
    elif cfg["fusion"] == "Noisy-Or":
        if 'rgbd' in mean:
            outputs = 1 - (1 - mean["rgb"]) * (1 - mean["d"]) * (1 - mean["rgbd"]) #[batch,11,512,512,1]
        else:
            outputs = 1 - (1 - mean["rgb"]) * (1 - mean["d"]) #[batch,11,512,512,1]
    return outputs 


def imbalance(mean,cfg,**kargs):
    if not cfg["imbalance"]:
        return mean
    outputs = {}
    for m in cfg["models"].keys():
        prior = torch.tensor(1/np.exp(kargs['log_prior'])).to('cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
        prior[prior == float("inf")] = 0
        # outputs = torch.nn.Softmax(dim=1)(mean["rgb"]) * torch.nn.Softmax(dim=1)(mean["d"])
        # outputs = outputs/outputs.sum(1).unsqueeze(1)
        #rebalance_prior = [1/23]*19
        #rebalance_prior = np.array(rebalance_prior)
        #rebalance_prior[13:15] = 0
        #rebalance_prior[16:] = 0
        #rebalance_prior[10] = 10/23
        #outputs_temp = outputs*prior*torch.tensor(rebalance_prior).to('cuda').unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # mean[m] = torch.nn.Softmax(dim=1)(mean[m])
        mean_temp = mean[m]*prior
        mean_temp = mean_temp/mean_temp.sum(1).unsqueeze(1)
        mean_temp = mean[m]**cfg['beta'] * mean_temp**(1-cfg['beta']) # why multiplication ????
        outputs[m] = mean_temp/mean_temp.sum(1).unsqueeze(1)
    return outputs