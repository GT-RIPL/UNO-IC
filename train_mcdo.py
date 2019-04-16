import matplotlib
matplotlib.use('Agg')

import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter
from functools import partial

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
plt.ioff()

def tensor_hook(data,grad):
    output, cross_loss = data

    # sigma = output[:,int(output.shape[1]/2):,:,:]
    # grad_mu = grad[:,:int(grad.shape[1]/2),:,:]
    # grad_var = grad[:,int(grad.shape[1]/2):,:,:]

    # modified_grad = torch.cat((0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma,grad_var),1)

    sigma = torch.cat((output[:,int(output.shape[1]/2):,:,:],output[:,int(output.shape[1]/2):,:,:]),1)
    grad_mu = torch.cat((grad[:,:int(grad.shape[1]/2),:,:],grad[:,:int(grad.shape[1]/2),:,:]),1)

    # loss = (0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma).sum(-1).sum(-1).sum(-1)

    # print(loss.shape)


    # loss = loss.repeat(1,grad.shape[1],grad.shape[2],grad.shape[3])

    # print(grad.shape)
    # print(loss.shape)

    # print(output.shape,sigma.shape,grad.shape,grad_mu.shape)

    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*torch.mul(torch.exp(-sigma),grad.pow(2))+0.5*sigma
    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*sigma
    # modified_grad = torch.sum(0.5*torch.mul(torch.exp(-sigma),cross_loss)+0.5*sigma,dim=1)
    # modified_grad = 0.5*torch.mul(torch.exp(-sigma),grad)+0.5*sigma
    modified_grad = 0.5*torch.mul(torch.exp(-sigma),grad_mu)+0.5*sigma


    # only apply grad to mean, not also to std


    # modified_grad = loss

    return modified_grad


class Calibrator():
    def __init__(self,device):

        self.device = device

        self.W = torch.ones(1,device=device)
        self.b = torch.zeros(1,device=device)

    def fit(self,x_init,y_init,device):

        self.device = device

        self.W = torch.ones(len(x_init),device=device)
        self.b = torch.zeros(len(x_init),device=device)

        W = [0]*len(x_init)
        b = [0]*len(x_init)

        x_init, y_init = zip(*sorted(zip(x_init,y_init)))
        x_init = list(x_init)
        y_init = list(y_init)

        XX = zip(x_init[:-1],x_init[1:])
        YY = zip(y_init[:-1],y_init[1:])

        for i,XY in enumerate(zip(XX,YY)):
            X,Y = XY
            x1,x2 = X
            y1,y2 = Y
            
            self.W[i] = 1.*(y2-y1)/(x2-x1)
            self.b[i] = y2-self.W[i]*x2


        self.W.to(device)
        self.b.to(device)



    def predict(self,x):

        self.W.to(x.device)
        self.b.to(x.device)

        i = (1.*len(self.W)*torch.clamp(x,min=0,max=1)).floor().long()-1

        return self.W[i]*x + self.b[i]


# class CalibrationNet(torch.nn.Module):
#     def __init__(self):
#         super(CalibrationNet, self).__init__()

#         # Learn Piecewise Linear Functions

#         self.pieces = 20
#         self.W = torch.nn.Parameter(data=torch.ones(self.pieces,1), requires_grad=True)
#         self.b = torch.nn.Parameter(data=torch.zeros(1,1), requires_grad=True)
#         # self.b = torch.nn.Parameter(data=torch.Tensor(self.pieces,1), requires_grad=True)

#         # self.W.data = 1
#         # self.b.data = 0
        
#     def forward(self, x):
        
#         b = torch.Tensor(self.pieces,1)
#         b[0,:] = self.b[0,0]

#         for p in range(self.pieces-1):
#             y = torch.abs(self.W[p,:])*(1.*(p+1)/self.pieces) + b[p,:]
#             b[p+1,:] = y - torch.abs(self.W[p+1,:])*(1.*(p+1)/self.pieces)

#         i = (1.*self.pieces*torch.clamp(x,min=0,max=1)).ceil().long()-1

#         # print(self.W.shape,self.b.shape,b.shape,x.shape)
#         # print((self.W[i,0]*x + b[i,0]).shape)
#         # print(self.W[i,0].shape,b[i,0].shape)

#         # return x + self.b[i,0]
#         return torch.abs(self.W[i,0])*x + b[i,0]

#         # X = torch.cat(tuple([x**i for i in range(20)]),1)
        
#         # return torch.matmul(X,self.W)

#     # # def __init__(self, n_feature, n_hidden, n_output):
#     # def __init__(self):
#     #     super(CalibrationNet, self).__init__()
#     #     n_feature = 1
#     #     n_hidden = 100
#     #     n_output = 1
#     #     self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#     #     self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
#     # def forward(self, x):
#     #     x = F.relu(self.hidden(x))      # activation function for hidden layer
#     #     x = self.predict(x)             # linear output
#     #     return x

def parseEightCameras(images,labels,aux,device):

    # Stack 8 Cameras into 1 for MCDO Dataset Testing
    images = torch.cat(images,0)
    labels = torch.cat(labels,0)
    aux = torch.cat(aux,0)

    images = images.to(device)
    labels = labels.to(device)

    if len(aux.shape)<len(images.shape):
        aux = aux.unsqueeze(1).to(device)
        depth = torch.cat((aux,aux,aux),1)
    else:
        aux = aux.to(device)
        depth = torch.cat((aux[:,0,:,:].unsqueeze(1),
                           aux[:,1,:,:].unsqueeze(1),
                           aux[:,2,:,:].unsqueeze(1)),1)

    fused = torch.cat((images,aux),1)

    rgb = torch.cat((images[:,0,:,:].unsqueeze(1),
                     images[:,1,:,:].unsqueeze(1),
                     images[:,2,:,:].unsqueeze(1)),1)

    inputs = {"rgb": rgb,
              "d": depth,
              "fused": fused}

    return inputs, labels

def runModel(models,calibration,inputs,device):

    reg = torch.zeros(1,device=device )

    model2input = [
                   ("input_fusion","fused"),
                   ("rgb_only","rgb"),
                   ("d_only","d"),
                   ("rgb_static","rgb"),
                   ("d_static","d"),
                   ("rgb","rgb"),
                   ("d","d"),
                   ("fuse","fuse"),
                  ]

    # Relevant Models
    m2i = [(m,mi) for m,mi in model2input if m in models.keys()]

    # Relevant Modes
    modes = [mode for mode in ['rgb','d'] if any([mode==m for m in models.keys()])]

    mean_outputs = {}; var_outputs = {}; var_soft_outputs = {}; var_recal = {}; regs = {}; outputs = {}; outputs_aux = {}


    ####################################################
    # Single Model: RGB Only, Depth Only, Input Fusion #
    ####################################################
    if len(m2i)==1:     
        model,input = m2i[0]  

        if int(cfg['models'][model]['mcdo_passes'])==1:
            outputs, _ = models[model](inputs[input])
        else:

            outputs = {}; outputs_aux = {}; regs = {}             

            # Set Number of MCDO Passes
            models[model].mcdo_passes = cfg['models'][model]['mcdo_passes']
            regs[model] = torch.zeros( models[model].mcdo_passes, device=device )

            # Perform One Forward Pass with Gradients
            x, regs[model][0] = models[model](inputs[input])
            x_aux = None
            if isinstance(x,tuple):
                x, x_aux = x
            outputs[model] = x.unsqueeze(-1)
            if not x_aux is None:
                outputs_aux[model] = x_aux.unsqueeze(-1)

            o = x
            oa = x_aux

            # And Remaining Forward Passes without Gradients
            with torch.no_grad():
                for mi in range(models[model].mcdo_passes-1):
                    x, regs[model][mi+1] = models[model](inputs[input])
                    x_aux = None
                    if isinstance(x,tuple):
                        x, x_aux = x
                    if not model in outputs:
                        outputs[model] = x.unsqueeze(-1)
                        if not x_aux is None:
                            outputs_aux[model] = x_aux.unsqueeze(-1)
                    else:
                        outputs[model] = torch.cat((outputs[model], x.unsqueeze(-1)),-1)
                        if not x_aux is None:
                            outputs_aux[model] = torch.cat((outputs_aux[model], x_aux.unsqueeze(-1)),-1)

            reg = torch.stack([regs[m].sum() for m in regs.keys()]).sum()

            # Calculate Statistics on Multiple Passes
            # mean_outputs = {}; var_outputs = {}
            for m in outputs.keys():
                mean_outputs[m] = outputs[m].mean(-1)
                if models[m].mcdo_passes>1:
                    var_outputs[m] = outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2)
                else:
                    var_outputs[m] = torch.ones(mean_outputs[m].shape,device=device) #mean_outputs[m]  

            if len(outputs_aux)>0:
                mean_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}
                with torch.no_grad():
                    if models[m].mcdo_passes>1:
                        var_outputs_aux = {m:outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2) for m in outputs_aux.keys()}
                    else:
                        var_outputs_aux = {m:torch.ones(mean_outputs[m].shape,device=device) for m in outputs_aux.keys()}

            # Convert Variances to Softmax Probabilities and Recalibrate
            # var_soft_outputs = {m:torch.nn.Softmax(1)(var_outputs[m]) for m in var_outputs.keys()}

            # Use Softmax Class Probability for Recalibration
            var_soft_outputs = {m:torch.nn.Softmax(1)(mean_outputs[m]) for m in outputs.keys()}

            # print(var_soft_outputs[m].shape)
            # print(var_soft_outputs[m].cpu().numpy().reshape(-1).shape)

            if reg.requires_grad == True or not calibration[m]['fit']:
                var_recal = {m:var_soft_outputs[m] for m in var_outputs.keys()}
            else:
                var_recal = {m:(calibration[m]['model'].predict(var_soft_outputs[m].reshape(-1)).reshape(var_soft_outputs[m].shape)) for m in var_outputs.keys()}

            # # UNCERTAINTY
            # # convert log variance to normal variance
            # for m in mean_outputs.keys():
            #     var_split = int(mean_outputs[m].shape[1]/2)
            #     mean_outputs[m][:,:var_split,:,:] = torch.exp(mean_outputs[m][:,:var_split,:,:])

            # auxiliaring training loss
            if len(outputs_aux)>0:
                # outputs = (outputs[model],*[mean_outputs_aux[m] for m in mean_outputs_aux.keys()])
                outputs = (o,oa)
            else:
                outputs = o



    ###################################################
    # Multiple Models: Middle Fusion, Isolated Fusion #
    ###################################################
    else:

        ####################################
        # Single Pass Prelude to MCDO Legs #
        ####################################
        outputs = {}
        outputs_aux = {}
        if any("_static" in m for m in models.keys()):                
            outputs = {m:list(models[m+"_static"](inputs[m]))[0] for m in modes}
            inputs = outputs

        ###########################
        # Multiple Pass MCDO Legs #
        ###########################
        outputs = {}; outputs_aux = {}; regs = {};
        o = {}; oa = {}           
        for m in modes:

            # Set Number of MCDO Passes
            models[m].mcdo_passes = cfg['models'][m]['mcdo_passes']
            regs[m] = torch.zeros( models[m].mcdo_passes, device=device )

            # Perform One Forward Pass with Gradients
            x, regs[m][0] = models[m](inputs[m])
            x_aux = None
            if isinstance(x,tuple):
                x, x_aux = x
            outputs[m] = x.unsqueeze(-1)
            if not x_aux is None:
                outputs_aux[m] = x_aux.unsqueeze(-1)

            o[m] = x
            oa[m] = x_aux

            # And Remaining Forward Passes without Gradients
            with torch.no_grad():
                for mi in range(models[m].mcdo_passes-1):
                    x, regs[m][mi+1] = models[m](inputs[m])
                    x_aux = None
                    if isinstance(x,tuple):
                        x, x_aux = x
                    if not m in outputs:
                        outputs[m] = x.unsqueeze(-1)
                        if not x_aux is None:
                            outputs_aux[m] = x_aux.unsqueeze(-1)
                    else:
                        outputs[m] = torch.cat((outputs[m], x.unsqueeze(-1)),-1)
                        if not x_aux is None:
                            outputs_aux[m] = torch.cat((outputs_aux[m], x_aux.unsqueeze(-1)),-1)

            reg = torch.stack([regs[m].sum() for m in regs.keys()]).sum()

        # Calculate Statistics on Multiple Passes
        # mean_outputs = {}; var_outputs = {}
        for m in outputs.keys():
            mean_outputs[m] = outputs[m].mean(-1)
            if models[m].mcdo_passes>1:
                var_outputs[m] = outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2)
            else:
                var_outputs[m] = torch.ones(mean_outputs[m].shape,device=device) #mean_outputs[m]  

        if len(outputs_aux)>0:
            mean_outputs_aux = {m:outputs_aux[m].mean(-1) for m in outputs_aux.keys()}
            with torch.no_grad():
                if models[m].mcdo_passes>1:
                    var_outputs_aux = {m:outputs[m].pow(2).mean(-1)-mean_outputs[m].pow(2) for m in outputs_aux.keys()}
                else:
                    var_outputs_aux = {m:torch.ones(mean_outputs[m].shape,device=device) for m in outputs_aux.keys()}

        # Convert Variances to Softmax Probabilities and Recalibrate
        # var_soft_outputs = {m:torch.nn.Softmax(1)(var_outputs[m]) for m in var_outputs.keys()}

        # Use Softmax Class Probability for Recalibration
        # var_soft_outputs = {m:torch.nn.Softmax(1)(mean_outputs[m]) for m in outputs.keys()}

        # Get Probability of Output Class from Gaussian Parameters

        for m in mean_outputs.keys():
            for i in range(mean_outputs[m].shape[1]):
                print("{}: {} {}".format(m,mean_outputs[m][0,i,0,0],var_outputs[m][0,i,0,0]))

        with torch.no_grad():
            G = torch.distributions.normal.Normal(0,1)
            var_soft_outputs = {}
            for m in outputs.keys():
                max_mu_i = torch.argmax(mean_outputs[m],dim=1)
                max_mu = mean_outputs[m].gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)
                max_sigma = var_outputs[m].gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)

                cum_cdf = torch.zeros(max_mu_i.shape,device=device)
                for x in np.arange(-1,1,0.1):
                    x_scale = max_mu + 5*max_sigma*x
                    step = 5*max_sigma*0.1

                    cdf = torch.ones(cum_cdf.shape,device=device)    
                    
                    for i in range(int(mean_outputs[m].shape[1])):
                        cdf = torch.mul( cdf, G.cdf((x_scale-mean_outputs[m][:,i,:,:])/var_outputs[m][:,i,:,:]) )

                    # cdf = torch.div( cdf, G.cdf((x_scale-mean_outputs[m].gather(1,max_mu_i.unsqueeze(1)).squeeze(1))/var_outputs[m].gather(1,max_mu_i.unsqueeze(1)).squeeze(1)) )

                    cdf = cdf * torch.exp(G.log_prob((x_scale-max_mu)/(max_sigma)))

                    cum_cdf = cum_cdf + cdf*step

                var_soft_outputs[m] = cum_cdf

        for m in mean_outputs.keys():
            print("{}: {} {}".format(m,mean_outputs[m].data.max(1)[1][0,0,0],var_soft_outputs[m][0,0,0]))
            # exit()


        if reg.requires_grad == True or not calibration[m]['fit']:
            var_recal = {m:var_soft_outputs[m] for m in var_outputs.keys()}
        else:
            # var_recal = {m:torch.from_numpy(calibration[m].predict(var_soft_outputs[m].cpu().numpy().reshape(-1)).reshape(var_soft_outputs[m].shape)).double() for m in var_outputs.keys()}
            var_recal = {m:(calibration[m]['model'].predict(var_soft_outputs[m].reshape(-1)).reshape(var_soft_outputs[m].shape)) for m in var_outputs.keys()}


        # # UNCERTAINTY
        # # convert log variance to normal variance
        # for m in mean_outputs.keys():
        #     var_split = int(mean_outputs[m].shape[1]/2)
        #     mean_outputs[m][:,:var_split,:,:] = torch.exp(mean_outputs[m][:,:var_split,:,:])


        # ###############################################
        # # Weight Fusion (Channel Stack, Weighted Sum) #
        # ###############################################
        # if cfg['models']['fuse']['in_channels'] == 0:
        #     # stack outputs from parallel legs
        #     intermediate = torch.cat(tuple([o[m] for m in outputs.keys()]+[var_outputs[m] for m in outputs.keys()]),1)
        # if cfg['models']['fuse']['in_channels'] == -1:
        #     if len(modes)==1:
        #         intermediate = o[list(o.keys())[0]]
        #     else:
        #         normalizer = var_outputs["rgb"] + var_outputs["d"]
        #         normalizer[normalizer==0] = 1
        #         intermediate = ((o["rgb"]*var_outputs["d"]) + (o["d"]*var_outputs["rgb"]))/normalizer


        ################
        # Fusion Trunk #
        ################
        outputs, _ = models['fuse']((mean_outputs,var_recal))

        # auxiliaring training loss
        if len(outputs_aux)>0:
            outputs = (outputs,*[oa[m] for m in mean_outputs_aux.keys()])

    return outputs, reg, (mean_outputs,var_soft_outputs,var_recal)


def fitCalibration(recalloader, models, calibration, n_classes, device):

    # Calibration Visualization
    steps = 10
    ranges = list(zip([1.*a/steps for a in range(steps+2)][:-2],
                      [1.*a/steps for a in range(steps+2)][1:]))

    val = ['sumval_pred_in_range',
           'num_obs_in_range',
           'num_in_range',
           'sumval_pred_below_range',
           'num_obs_below_range',
           'num_below_range',
           'num_correct']

    with torch.no_grad():

        # Evaluate Calibration
        print("Evaluating Calibration")
        for i_recal, (images_recal, labels_recal, aux_recal) in tqdm(enumerate(recalloader)):                            

            inputs_recal, labels_recal = parseEightCameras( images_recal, labels_recal, aux_recal, device )

            if labels_recal.shape[0]<=1:
                continue

            outputs_recal, reg_recal, meanvar_recal = runModel(models,calibration,inputs_recal,device)
            mean_outputs, var_soft_outputs, var_recal = meanvar_recal

            gt = labels_recal.data.cpu().numpy()

            per_class_match_var = {m:{r:{c:{v:0 for v in val} for c in range(n_classes)} for r in ranges} for m in var_soft_outputs.keys()}
            overall_match_var = {m:{r:{v:0 for v in val} for r in ranges} for m in var_soft_outputs.keys()}


            for m in var_soft_outputs.keys():


                # # try softmax confidences first
                # pred = outputs_recal.data.max(1)[1].cpu().numpy()
                # conf = outputs_recal.data.max(1)[0].cpu().numpy()
                # pred_var = conf.copy()

                # MCDO softmax
                # extract softmaxed variances for all output classes
                full = var_soft_outputs[m].cpu().numpy()
                
                # prediction is max of mean outputs
                pred = mean_outputs[m].data.max(1)[1]
                
                # extract the variance associated with the prediction
                # pred_var = var_soft_outputs[m].gather(1,pred.clone().unsqueeze(1)).squeeze(1)
                pred_var = var_soft_outputs[m]

                # convert to numpy
                pred = pred.cpu().numpy()
                pred_var = pred_var.cpu().numpy()


                # fig = plt.figure()
                # plt.subplot(2,1,1)
                # plt.imshow(pred[0,:,:])
                # plt.subplot(2,1,2)
                # plt.imshow(pred_var[0,:,:])
                # plt.show()

                # print(pred.shape)
                # print(pred_var.shape)


                


                for r in ranges:
                    # for each probability range
                    # (1) tally correct labels (classes) for empirical confidence 
                    # (2) average predicted confidences
                    low,high = r
                    idx_pred_gt_match = (pred==gt) # index of all correct labels
                    idx_pred_var_in_range = (low<=pred_var)&(pred_var<high) # index with specific variance
                    idx_pred_var_below_range = (pred_var<high) # index with specific variance

                    sumval_pred_var_in_range = np.sum(pred_var[idx_pred_var_in_range][:])
                    sumval_pred_var_below_range = np.sum(pred_var[idx_pred_var_below_range][:])

                    num_obs_var_in_range = np.sum((idx_pred_gt_match&idx_pred_var_in_range)[:])
                    num_obs_var_below_range = np.sum((idx_pred_gt_match&idx_pred_var_below_range)[:])

                    # sum_total = np.sum(idx_pred_var_in_range[:])
                    num_in_range = np.sum(idx_pred_var_in_range[:])
                    num_below_range = np.sum(idx_pred_var_below_range[:])
                    num_correct = np.sum(idx_pred_gt_match[:]) 

                    overall_match_var[m][r]['sumval_pred_in_range'] += sumval_pred_var_in_range
                    overall_match_var[m][r]['num_obs_in_range'] += num_obs_var_in_range
                    overall_match_var[m][r]['num_in_range'] += num_in_range

                    overall_match_var[m][r]['sumval_pred_below_range'] += sumval_pred_var_below_range
                    overall_match_var[m][r]['num_obs_below_range'] += num_obs_var_below_range
                    overall_match_var[m][r]['num_below_range'] += num_below_range

                    overall_match_var[m][r]['num_correct'] += num_correct


                    # for c in range(n_classes):
                    #     # for each class, record number of correct labels for each confidence bin
                    #     # for each class, record average confidence for each confidence bin 

                    #     low,high = r
                    #     idx_pred_gt_match = (pred==gt)&(pred==c) # everywhere correctly labeled to correct class
                    #     idx_pred_var_in_range = (low<=full[:,c,:,:])&(full[:,c,:,:]<high) # everywhere with specified confidence level
                    #     idx_pred_var_below_range = (full[:,c,:,:]<high) # everywhere with specified confidence level

                    #     sumval_pred_var_in_range = np.sum(pred_var[idx_pred_var_in_range][:])
                    #     sumval_pred_var_below_range = np.sum(pred_var[idx_pred_var_below_range][:])

                    #     num_obs_var_in_range = np.sum((idx_pred_gt_match&idx_pred_var_in_range)[:])
                    #     num_obs_var_below_range = np.sum((idx_pred_gt_match&idx_pred_var_below_range)[:])

                    #     num_in_range = np.sum(idx_pred_var_in_range[:])
                    #     num_below_range = np.sum(idx_pred_var_below_range[:])
                    #     num_correct = np.sum(idx_pred_gt_match[:]) 

                    #     per_class_match_var[m][r][c]['sumval_pred_in_range'] += sumval_pred_var_below_range
                    #     per_class_match_var[m][r][c]['num_obs_in_range'] += num_obs_var_in_range
                    #     per_class_match_var[m][r][c]['num_in_range'] += num_in_range

                    #     per_class_match_var[m][r][c]['sumval_pred_below_range'] += sumval_pred_var_below_range
                    #     per_class_match_var[m][r][c]['num_obs_below_range'] += num_obs_var_below_range
                    #     per_class_match_var[m][r][c]['num_below_range'] += num_below_range

                    #     per_class_match_var[m][r][c]['num_correct'] += num_correct



    for m in overall_match_var.keys():

        for r in ranges:
            low,high = r

            # den = overall_match_var[m][r]['num_correct']
            # den = den if den>0 else 1
            # overall_match_var[m][r]['pred'] = 1.*overall_match_var[m][r]['sumval_pred_in_range']/den #overall_match_var[m][r]['num_in_range']
            # overall_match_var[m][r]['obs'] = 1.*overall_match_var[m][r]['num_obs_in_range']/den #overall_match_var[m][r]['num_in_range']

            den = overall_match_var[m][r]['num_in_range']
            den = den if den>0 else 1
            overall_match_var[m][r]['pred'] = 1.*overall_match_var[m][r]['sumval_pred_in_range']/den #overall_match_var[m][r]['num_in_range']
            overall_match_var[m][r]['obs'] = 1.*overall_match_var[m][r]['num_obs_in_range']/den #overall_match_var[m][r]['num_in_range']

            # den = overall_match_var[m][r]['num_correct']
            # den = den if den>0 else 1
            # overall_match_var[m][r]['pred_below'] = high
            # overall_match_var[m][r]['obs_below'] = 1.*overall_match_var[m][r]['num_obs_below_range']/den

            den = overall_match_var[m][r]['num_below_range']
            den = den if den>0 else 1
            overall_match_var[m][r]['pred_below'] = 1.*overall_match_var[m][r]['sumval_pred_below_range']/den #overall_match_var[m][r]['num_in_range']
            overall_match_var[m][r]['obs_below'] = 1.*overall_match_var[m][r]['num_obs_below_range']/den #overall_match_var[m][r]['num_in_range']

            # for c in range(n_classes):
            #     # den = per_class_match_var[m][r][c]['num_correct']
            #     # den = den if den>0 else 1
            #     # per_class_match_var[m][r][c]['pred'] = 1.*per_class_match_var[m][r][c]['sumval_pred_in_range']/den #per_class_match_var[m][r][c]['num_in_range']
            #     # per_class_match_var[m][r][c]['obs'] = 1.*per_class_match_var[m][r][c]['num_obs_in_range']/den #per_class_match_var[m][r][c]['num_in_range']           

            #     den = per_class_match_var[m][r][c]['num_in_range']
            #     den = den if den>0 else 1
            #     per_class_match_var[m][r][c]['pred'] = 1.*per_class_match_var[m][r][c]['sumval_pred_in_range']/den #per_class_match_var[m][r][c]['num_in_range']
            #     per_class_match_var[m][r][c]['obs'] = 1.*per_class_match_var[m][r][c]['num_obs_in_range']/den #per_class_match_var[m][r][c]['num_in_range']

            #     # den = per_class_match_var[m][r][c]['num_correct']
            #     # den = den if den>0 else 1
            #     # per_class_match_var[m][r][c]['pred_below'] = high
            #     # per_class_match_var[m][r][c]['obs_below'] = 1.*per_class_match_var[m][r][c]['num_obs_below_range']/den #per_class_match_var[m][r][c]['num_in_range']           

            #     den = per_class_match_var[m][r][c]['num_below_range']
            #     den = den if den>0 else 1
            #     per_class_match_var[m][r][c]['pred_below'] = 1.*per_class_match_var[m][r][c]['sumval_pred_below_range']/den #per_class_match_var[m][r][c]['num_in_range']
            #     per_class_match_var[m][r][c]['obs_below'] = 1.*per_class_match_var[m][r][c]['num_obs_below_range']/den #overall_match_var[m][r]['num_in_range']



        x = np.array([overall_match_var[m][r]['pred'] for r in ranges])
        y = np.array([overall_match_var[m][r]['obs'] for r in ranges])
        # x = np.array([overall_match_var[m][r]['pred_below'] for r in ranges])
        # y = np.array([overall_match_var[m][r]['obs_below'] for r in ranges])

        # xp = np.arange(0,1,0.01)
        # yp = np.interp(xp,x,y)
        # x = np.array(xp)
        # y = np.array(yp)

        x = torch.from_numpy(x.reshape(-1,1)).float()
        y = torch.from_numpy(y).float()

        # Fit Calibration if Not Already
        if not calibration[m]['fit']:
            calibration[m]['model'].fit(x,y,device)
            calibration[m]['fit'] = True # True



        # for t in range(1000):
        #     if t%100==0:
        #         print("Training Recalibration: {}".format(t))
        #     prediction = calibration[m]["model"](x)     # input x and predict based on x
        #     loss = calibration[m]['loss_func'](prediction, y)     # must be (1. nn output, 2. target)

        #     calibration[m]["optim"].zero_grad()   # clear gradients for next train
        #     loss.backward()         # backpropagation, compute gradients
        #     calibration[m]["optim"].step()        # apply gradients

        # calibration[m].fit_transform(x,y)
        # calibration[m].fit(x[:,np.newaxis],y)

    return ranges, overall_match_var, per_class_match_var, calibration


def showCalibration(ranges, overall_match_var, per_class_match_var, calibration, logdir, i, n_classes, device, split):

    with torch.no_grad():

        for m in overall_match_var.keys():

            ###########
            # Overall #
            ###########
            fig, axes = plt.subplots(1,3)
            # [axi.set_axis_off() for axi in axes.ravel()]

            # Plot Predicted Variance Against Observed/Empirical Variance
            # x = [overall_match_var[m][r]['pred_below'] for r in ranges]
            # y = [overall_match_var[m][r]['obs_below'] for r in ranges]
            x = [overall_match_var[m][r]['pred'] for r in ranges]
            y = [overall_match_var[m][r]['obs'] for r in ranges]

            xp = np.arange(0,1,0.001)
            yp = np.interp(xp,x,y)


            # steps = 20
            # xp = []; yp = []
            # for t in range(100):
            #     xp += list([x[i]+1.*t/(steps*100) for i,r in enumerate(ranges)])
            #     yp += list(y)

            x = xp; y = yp

            axes[0].plot(x,y,'.')
            axes[0].set_title("Uncalibrated")

            # calibration['model'].eval()
            # y_recal = calibration['model'](torch.tensor(x).view(-1,1)).cpu().numpy()
            # axes[1].plot(x,y_recal)


            # Convert Predicted Variances to Calibrated Variances
            # x = np.array([overall_match_var[m][r]['pred_below'] for r in ranges])
            # y = np.array([overall_match_var[m][r]['obs_below'] for r in ranges])
            x = np.array([overall_match_var[m][r]['pred'] for r in ranges])
            y = np.array([overall_match_var[m][r]['obs'] for r in ranges])

            x = torch.from_numpy(x.reshape(-1,1)).float()

            # y_pred = calibration[m]["model"](x)
            y_pred = calibration[m]["model"].predict(x.to(device))
            y_pred = y_pred.cpu().numpy()

            # y_pred = calibration[m].predict(x)
            # y_pred = calibration[m].predict(x[:,np.newaxis])
            axes[1].plot(y,y_pred)                    
            axes[1].set_title("Recalibrated")

            # Recalibration Curve
            x = np.arange(0,1,0.001)
            x = torch.from_numpy(x.reshape(-1,1)).float()

            # y = calibration[m]["model"](x)
            y = calibration[m]["model"].predict(x.to(device))

            x = x.cpu().numpy()
            y = y.cpu().numpy()

            # y = calibration[m].predict(x)
            # y = calibration[m].predict(x[:,np.newaxis])
            axes[2].plot(x,y)                    
            axes[2].set_title("Recalibration Curve")

            path = "{}/{}/{}/{}".format(logdir,'calibration',split,m)
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig("{}/calibOverall{}.png".format(path,i))
            plt.close(fig)


            # ###############
            # # All Classes #
            # ###############
            # fig, axes = plt.subplots(3,n_classes//3+1)
            # # [axi.set_axis_off() for axi in axes.ravel()]

            # # x = [overall_match_var[m][r]['pred_below'] for r in ranges]
            # # y = [overall_match_var[m][r]['obs_below'] for r in ranges]
            # x = [overall_match_var[m][r]['pred'] for r in ranges]
            # y = [overall_match_var[m][r]['obs'] for r in ranges]
            # axes[0,0].plot(x,y)

            # for c in range(n_classes):
            #     # x = [per_class_match_var[m][r][c]['pred_below'] for r in ranges]
            #     # y = [per_class_match_var[m][r][c]['obs_below'] for r in ranges]                        
            #     x = [per_class_match_var[m][r][c]['pred'] for r in ranges]
            #     y = [per_class_match_var[m][r][c]['obs'] for r in ranges]                                        
            #     axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].plot(x,y)
            #     axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].set_title("Class: {}".format(c))

            # path = "{}/{}/{}/{}".format(logdir,'calibration',split,m)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # plt.savefig("{}/calib{}.png".format(path,i))
            # plt.close(fig)

            print(overall_match_var[m])    


def train(cfg, writer, logger, logdir):
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=cfg['data']['train_reduction'],
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    r_loader = data_loader(
        data_path,
        is_transform=True,
        split="recal",
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=0.25,
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    tv_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        subsplits=cfg['data']['train_subsplit'],
        scale_quantity=0.05,
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = {env:data_loader(
        data_path,
        is_transform=True,
        split="val", subsplits=[env], scale_quantity=cfg['data']['val_reduction'],
        img_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),) for env in cfg['data']['val_subsplit']}

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    recalloader = data.DataLoader(r_loader,
                                  batch_size=1, #cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloaders = {key:data.DataLoader(v_loader[key], 
                                      batch_size=cfg['training']['batch_size'], 
                                      num_workers=cfg['training']['n_workers']) for key in v_loader.keys()}

    # add training samples to validation sweep
    valloaders = {**valloaders,'train':data.DataLoader(tv_loader,
                                                       batch_size=cfg['training']['batch_size'], 
                                                       num_workers=cfg['training']['n_workers'])}

    # Setup Metrics
    running_metrics_val = {env:runningScore(n_classes) for env in valloaders.keys()}
    val_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    val_CE_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    val_REG_loss_meter = {env:averageMeter() for env in valloaders.keys()}
    

    start_iter = 0
    models = {}
    optimizers = {}
    schedulers = {}

    layers = [  "convbnrelu1_1",
                "convbnrelu1_2",
                "convbnrelu1_3",
                   "res_block2",
                   "res_block3",
                   "res_block4",
                   "res_block5",
              "pyramid_pooling",
                    "cbr_final",
               "classification"]


    # Create Isotonic Regression Model for Each MCDO Branch
    # Initially Fit to Identity Transfer Function
    calibration = {}
    for m in cfg["models"].keys():
        if cfg["models"][m]["mcdo_passes"]>=1:
            x = np.arange(0,1,0.01)

            # calibration[m] = IsotonicRegression()
            # calibration[m] = LinearRegression()
            # calibration[m].fit_transform(x,x)
            # calibration[m].fit(x[:,np.newaxis],x)

            calibration[m] = {}
            calibration[m]['model'] = Calibrator(device) #n_feature=1, n_hidden=40, n_output=1)
            # calibration[m]['model'] = CalibrationNet() #n_feature=1, n_hidden=40, n_output=1)
            # print(list(calibration[m]['model'].parameters()))
            # calibration[m]['loss_func'] = nn.MSELoss()
            # calibration[m]['optim'] = torch.optim.SGD(calibration[m]['model'].parameters(),lr=0.001)
            calibration[m]['fit'] = False

    # calibration['model'] = IsotonicRegression()
    # calibration['fit'] = False
    
    # calibration['model'] = nn.Linear(1,1)
    # calibration['criterion'] = nn.MSELoss()
    # l_rate = 0.01
    # calibration['optim'] = torch.optim.SGD(calibration['model'].parameters(),lr=l_rate)

    for model,attr in cfg["models"].items():
        if len(cfg['models'])==1:
            start_layer = "convbnrelu1_1"
            end_layer = "classification"
        else:  
            if not str(cfg['start_layers']) == "None":
                if "_static" in model:
                    start_layer = "convbnrelu1_1"
                    end_layer = layers[layers.index(cfg['start_layers'][1])-1]
                elif "fuse" == model:
                    start_layer = cfg['start_layers'][-1]
                    end_layer = "classification"
                else:
                    if len(cfg['start_layers']) == 3:
                        start_layer = cfg['start_layers'][1]
                        end_layer = layers[layers.index(cfg['start_layers'][2])-1]
                    else:
                        start_layer = cfg['start_layers'][0]
                        end_layer = layers[layers.index(cfg['start_layers'][1])-1]
            else:
                start_layer = attr['start_layer']
                end_layer = attr['end_layer']

        print(model,start_layer,end_layer)

        models[model] = get_model(cfg['model'], 
                                  n_classes, 
                                  input_size=(cfg['data']['img_rows'],cfg['data']['img_cols']),
                                  in_channels=attr['in_channels'],
                                  start_layer=start_layer,
                                  end_layer=end_layer,
                                  mcdo_passes=attr['mcdo_passes'], 
                                  dropoutP=attr['dropoutP'],
                                  learned_uncertainty=attr['learned_uncertainty'],
                                  reduction=attr['reduction']).to(device)

        # # Load Pretrained PSPNet
        # if cfg['model'] == 'pspnet':
        #     caffemodel_dir_path = "./models"
        #     model.load_pretrained_model(
        #         model_path=os.path.join(caffemodel_dir_path, "pspnet101_cityscapes.caffemodel")
        #     )  

        if "caffemodel" in attr['resume']:
            models[model].load_pretrained_model(model_path=attr['resume'])


        models[model] = torch.nn.DataParallel(models[model], device_ids=range(torch.cuda.device_count()))

        # Setup optimizer, lr_scheduler and loss function
        optimizer_cls = get_optimizer(cfg)
        optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                            if k != 'name'}

        optimizers[model] = optimizer_cls(models[model].parameters(), **optimizer_params)
        logger.info("Using optimizer {}".format(optimizers[model]))

        schedulers[model] = get_scheduler(optimizers[model], cfg['training']['lr_schedule'])

        loss_fn = get_loss_function(cfg)
        # loss_sig = # Loss Function for Aleatoric Uncertainty
        logger.info("Using loss {}".format(loss_fn))

        # Load pretrained weights
        if attr['resume'] is not None and not "caffemodel" in attr['resume']:

            model_pkl = attr['resume']
            if attr['resume']=='same_yaml':
                model_pkl = "{}/{}_pspnet_airsim_best_model.pkl".format(logdir,model)

            if os.path.isfile(model_pkl):
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(model_pkl)
                )
                checkpoint = torch.load(model_pkl)

                ###
                pretrained_dict = torch.load(model_pkl)['model_state']
                model_dict = models[model].state_dict()

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v.resize_(model_dict[k].shape) for k, v in pretrained_dict.items() if (k in model_dict)} # and ((model!="fuse") or (model=="fuse" and not start_layer in k))}

                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 

                # 3. load the new state dict
                models[model].load_state_dict(pretrained_dict)

                if attr['resume']=='same_yaml':
                    # models[model].load_state_dict(checkpoint["model_state"])
                    optimizers[model].load_state_dict(checkpoint["optimizer_state"])
                    schedulers[model].load_state_dict(checkpoint["scheduler_state"])
                    start_iter = checkpoint["epoch"]
                else:
                    start_iter = 0

                # start_iter = 0
                logger.info("Loaded checkpoint '{}' (iter {})".format(model_pkl, checkpoint["epoch"]))
            else:
                logger.info("No checkpoint found at '{}'".format(model_pkl))        

        # val_loss_meter[model] = averageMeter()


    # val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, aux) in trainloader:
            i += 1
            start_ts = time.time()

            inputs, labels = parseEightCameras( images, labels, aux, device )

            if labels.shape[0]<=1:
                continue

            [schedulers[m].step() for m in schedulers.keys()]
            [models[m].train() for m in models.keys()]
            [optimizers[m].zero_grad() for m in optimizers.keys()]

            outputs, reg, mean_var_outputs = runModel(models,calibration,inputs,device)
            mean_outputs, var_soft_outputs, var_recal = mean_var_outputs

            # if calibration['fit']:
            #     # var_soft_outputs = {m:torch.nn.Softmax(1)(var_outputs[m]) for m in var_outputs.keys()}
            #     var_recal = calibration['model'](var_soft_outputs)


            # with torch.no_grad():
            #     print(mean_outputs['rgb_only'].cpu().numpy())
            #     print(var_outputs['rgb_only'].cpu().numpy())
            #     # print(intermediate.cpu().numpy().shape)
            #     print(outputs.cpu().numpy())
     
            CE_loss = loss_fn(input=outputs,target=labels)
            REG_loss = reg
            loss = CE_loss + 1e6*REG_loss


            # register hooks for modifying gradients for learned uncertainty
            if len(cfg['models'])>1 and cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                hooks = {m:mean_outputs[m].register_hook(partial(tensor_hook,(mean_outputs[m],loss))) for m in mean_outputs.keys()}

            loss.backward()

            # remove hooks for modifying gradients for learned uncertainty
            if len(cfg['models'])>1 and cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                [hooks[h].remove() for h in hooks.keys()]            

            [optimizers[m].step() for m in optimizers.keys()]





            time_meter.update(time.time() - start_ts)
            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i+1)
                writer.add_scalar('loss/train_CE_loss', CE_loss.item(), i+1)
                writer.add_scalar('loss/train_REG_loss', REG_loss, i+1)
                time_meter.reset()








            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                
                [models[m].eval() for m in models.keys()]

                # Recalibration
                mcdo_model_name = "rgb" if len(cfg['models'])>1 else list(cfg['models'].keys())[0] #next((s for s in list(cfg['models'].keys()) if "mcdo" in s), None)
                if cfg['models'][mcdo_model_name]['mcdo_passes']>=1:
                    # with torch.no_grad():



                        # Recalibration Set
                        ranges, \
                        overall_match_var, \
                        per_class_match_var, \
                        calibration = fitCalibration(recalloader, 
                                                     models, 
                                                     calibration, 
                                                     n_classes, 
                                                     device)

                        showCalibration(ranges,
                                        overall_match_var, 
                                        per_class_match_var, 
                                        calibration, 
                                        logdir, 
                                        i, 
                                        n_classes,
                                        device,
                                        "recal")

                        # Validation Set
                        for k,valloader in valloaders.items():
                            ranges, \
                            overall_match_var, \
                            per_class_match_var, \
                            calibration = fitCalibration(valloader, 
                                                         models, 
                                                         calibration, 
                                                         n_classes, 
                                                         device)

                            showCalibration(ranges,
                                            overall_match_var, 
                                            per_class_match_var, 
                                            calibration, 
                                            logdir, 
                                            i, 
                                            n_classes,
                                            device,
                                            k)


                exit()

                # Validation Set
                with torch.no_grad():

                    for k,valloader in valloaders.items():
                        for i_val, (images, labels, aux) in tqdm(enumerate(valloader)):
                            
                            inputs, labels = parseEightCameras( images, labels, aux, device )

                            orig = inputs.copy()

                            if labels.shape[0]<=1:
                                continue

                            outputs, reg, _ = runModel(models,calibration,inputs,device)
     
                            CE_loss = loss_fn(input=outputs,target=labels)
                            REG_loss = reg
                            val_loss = CE_loss + 1e6*REG_loss

                            pred = outputs.data.max(1)[1].cpu().numpy()
                            conf = outputs.data.max(1)[0].cpu().numpy()
                            gt = labels.data.cpu().numpy()



                            # # Visualization
                            # if i_val % cfg['training']['png_frames'] == 0:
                            #     fig, axes = plt.subplots(3,4)
                            #     [axi.set_axis_off() for axi in axes.ravel()]

                            #     gt_norm = gt[0,:,:].copy()
                            #     pred_norm = pred[0,:,:].copy()

                            #     gt_norm[0,0] = 0
                            #     gt_norm[0,1] = n_classes
                            #     pred_norm[0,0] = 0
                            #     pred_norm[0,1] = n_classes

                            #     axes[0,0].imshow(gt_norm)
                            #     axes[0,0].set_title("GT")

                            #     axes[0,1].imshow(orig['rgb'][0,:,:,:].permute(1,2,0).cpu().numpy()[:,:,0])
                            #     axes[0,1].set_title("RGB")

                            #     axes[0,2].imshow(orig['d'][0,:,:,:].permute(1,2,0).cpu().numpy())
                            #     axes[0,2].set_title("D")

                            #     axes[1,0].imshow(pred_norm)
                            #     axes[1,0].set_title("Pred")

                            #     axes[2,0].imshow(conf[0,:,:])
                            #     axes[2,0].set_title("Conf")


                            #     if len(cfg['models'])>1:
                            #         if cfg['models']['rgb']['learned_uncertainty'] == 'yes':            
                            #             channels = int(mean_outputs['rgb'].shape[1]/2)

                            #             axes[1,1].imshow(mean_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             axes[1,1].set_title("Aleatoric (RGB)")

                            #             axes[1,2].imshow(mean_outputs['d'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             # axes[1,2].imshow(mean_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             axes[1,2].set_title("Aleatoric (D)")

                            #         else:
                            #             channels = int(mean_outputs['rgb'].shape[1])

                            #         if cfg['models']['rgb']['mcdo_passes']>1:
                            #             axes[2,1].imshow(var_outputs['rgb'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             axes[2,1].set_title("Epistemic (RGB)")

                            #             axes[2,2].imshow(var_outputs['d'][:,:channels,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             # axes[2,2].imshow(var_outputs['rgb'][:,channels:,:,:].mean(1)[0,:,:].cpu().numpy())
                            #             axes[2,2].set_title("Epistemic (D)")


                                    

                            #     path = "{}/{}".format(logdir,k)
                            #     if not os.path.exists(path):
                            #         os.makedirs(path)
                            #     plt.savefig("{}/{}_{}.png".format(path,i_val,i))
                            #     plt.close(fig)

                            running_metrics_val[k].update(gt, pred)

                            val_loss_meter[k].update(val_loss.item())
                            val_CE_loss_meter[k].update(CE_loss.item())
                            val_REG_loss_meter[k].update(REG_loss)


                    for k in valloaders.keys():
                        writer.add_scalar('loss/val_loss/{}'.format(k), val_loss_meter[k].avg, i+1)
                        writer.add_scalar('loss/val_CE_loss/{}'.format(k), val_CE_loss_meter[k].avg, i+1)
                        writer.add_scalar('loss/val_REG_loss/{}'.format(k), val_REG_loss_meter[k].avg, i+1)
                        logger.info("%s Iter %d Loss: %.4f" % (k, i + 1, val_loss_meter[k].avg))
                
                for env,valloader in valloaders.items():
                    score, class_iou = running_metrics_val[env].get_scores()
                    for k, v in score.items():
                        print(k, v)
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/{}'.format(env,k), v, i+1)

                    for k, v in class_iou.items():
                        logger.info('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}/cls_{}'.format(env,k), v, i+1)

                    val_loss_meter[env].reset()
                    running_metrics_val[env].reset()

                for m in optimizers.keys():
                    model = models[m]
                    optimizer = optimizers[m]
                    scheduler = schedulers[m]

                    if score["Mean IoU : \t"] >= best_iou:
                        best_iou = score["Mean IoU : \t"]
                        state = {
                            "epoch": i + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "best_iou": best_iou,
                        }
                        save_path = os.path.join(writer.file_writer.get_logdir(),
                                                 "{}_{}_{}_best_model.pkl".format(
                                                     m,
                                                     cfg['model']['arch'],
                                                     cfg['data']['dataset']))
                        torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/pspnet_airsim.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    mcdo_model_name = "rgb" if len(cfg['models'])>1 else list(cfg['models'].keys())[0] #next((s for s in list(cfg['models'].keys()) if "mcdo" in s), None)

    name = [cfg['id']]
    name.append("{}x{}".format(cfg['data']['img_rows'],cfg['data']['img_cols']))
    name.append("{}Mode".format("-".join([mode for mode in ['rgb','d'] if any([mode==m for m in cfg['models'].keys()])])))
    name.append("_{}_".format("-".join(cfg['start_layers']) if len(cfg['models'])>1 else mcdo_model_name))
    # name.append("_{}_".format("-".join(cfg['start_layers'])))
    name.append("{}bs".format(cfg['training']['batch_size']))
    # name.append("_{}_".format("-".join(cfg['models'].keys())))
    if not mcdo_model_name is None:
        name.append("{}reduction".format(cfg['models'][mcdo_model_name]['reduction']))
        name.append("{}passes".format(cfg['models'][mcdo_model_name]['mcdo_passes']))
        name.append("{}dropoutP".format(cfg['models'][mcdo_model_name]['dropoutP']))
        name.append("{}learnedUncertainty".format(cfg['models'][mcdo_model_name]['learned_uncertainty']))
        name.append("{}mcdostart".format(cfg['models'][mcdo_model_name]['mcdo_start_iter']))
        name.append("{}mcdobackprop".format(cfg['models'][mcdo_model_name]['mcdo_backprop']))
        name.append("pretrain" if not str(cfg['models'][mcdo_model_name]['resume'])=="None" else "fromscratch")

    if any(["fuse"==m for m in cfg['models'].keys()]):
        name.append("{}".format("StackedFuse" if cfg['models']['fuse']['in_channels']==0 else "MultipliedFuse"))


    # name.append("_train_{}_".format(list(cfg['data']['train_subsplit'])[-1]))
    # name.append("_test_all_")
    name.append("01-16-2019")

    run_id = "_".join(name)

    # run_id = "mcdo_1pass_pretrain_alignedclasses_fused_fog_all_01-16-2019" #random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger, logdir)
