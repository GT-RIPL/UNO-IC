import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
       
def likelihood_flattening(mean, cfg, entropy, entropy_stats, modality):
    if not cfg['uncertainty']:
        return mean
    else:
        if modality == 'rgb': 
            SoftEn_MEAN = entropy_stats['rgb_mean']
            SoftEn_STD =  entropy_stats['rgb_std']
        else:
            SoftEn_MEAN = entropy_stats['d_mean']
            SoftEn_STD = entropy_stats['d_std']
        STD_MEAN = torch.max(torch.zeros_like(entropy.mean((1,2))),entropy.mean((1,2)) - SoftEn_MEAN - SoftEn_STD)+SoftEn_MEAN
        DR = SoftEn_MEAN/STD_MEAN.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return mean*DR
        


def fusion(mean,cfg,**kargs):
    if len(cfg['models'].keys()) == 1:
        return mean[list(cfg["models"].keys())[0]]
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
            outputs = 1 - (1 - mean["rgb"]) * (1 - mean["d"]) * (1 - mean["rgbd"]) #[batch,11,512,512]
        else:
            outputs = 1 - (1 - mean["rgb"]) * (1 - mean["d"]) #[batch,11,512,512]
    outputs = outputs/outputs.sum(1).unsqueeze(1)
    return outputs 


def prior_recbalancing(mean,cfg,**kargs):
    for m in cfg['models'].keys():
        mean[m] = torch.nn.Softmax(dim=1)(mean[m]) 
    if cfg["imbalance"]['beta'] is None:
        return mean

    inv_prior = 1/kargs['prior']
    inv_prior[inv_prior == float("inf")] = 0
    outputs = {}
    for m in cfg["models"].keys():
        mean_temp = mean[m]*inv_prior
        mean_temp = mean_temp/mean_temp.sum(1).unsqueeze(1)
        mean_temp = mean[m]**(1-cfg["imbalance"]['beta']) * mean_temp**cfg["imbalance"]['beta'] 
        outputs[m] = mean_temp/mean_temp.sum(1).unsqueeze(1)
    return outputs