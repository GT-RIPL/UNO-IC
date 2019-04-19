import torch.nn as nn

from ptsemseg.models.utils import *

import numpy as np
import matplotlib.pyplot as plt
import os

class Recalibrator():
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

def accumulateEmpirical(overall_match_var,ranges,label,mean,variance):

    softmax_mu = torch.nn.Softmax(1)(mean)
    variance = softmax_mu

    max_mu_i = torch.argmax(mean,dim=1)
    max_mu = mean.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)
    max_sigma = variance.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)


    # Define Ground Truth, Prediction, and Associated Prediction Probability
    pred = max_mu_i
    gt = label
    pred_var = max_sigma #softmax_mu
    

    m = 'rgb'

    for r in ranges:
        # for each probability range
        # (1) tally correct labels (classes) for empirical confidence 
        # (2) average predicted confidences
        low,high = r
        idx_pred_gt_match = (pred==gt) # index of all correct labels
        idx_pred_var_in_range = (low<=pred_var)&(pred_var<high) # index with specific variance
        idx_pred_var_below_range = (pred_var<high) # index with specific variance

        sumval_pred_var_in_range = torch.sum(pred_var[idx_pred_var_in_range])
        sumval_pred_var_below_range = torch.sum(pred_var[idx_pred_var_below_range])

        num_obs_var_in_range = torch.sum((idx_pred_gt_match&idx_pred_var_in_range))
        num_obs_var_below_range = torch.sum((idx_pred_gt_match&idx_pred_var_below_range))

        # sum_total = torch.sum(idx_pred_var_in_range)
        num_in_range = torch.sum(idx_pred_var_in_range)
        num_below_range = torch.sum(idx_pred_var_below_range)
        num_correct = torch.sum(idx_pred_gt_match)

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



    return overall_match_var

def fitCalibration(calibration,overall_match_var,ranges,device):

    m = 'rgb'

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



    x = np.array([overall_match_var[m][r]['pred'].cpu().numpy() for r in ranges])
    y = np.array([overall_match_var[m][r]['obs'].cpu().numpy() for r in ranges])
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


    return calibration, overall_match_var


def showCalibration(calibration,overall_match_var,ranges,logdir,cfg,n_classes,i,i_recal,device):

    for m in ["rgb"]:

        ###########
        # Overall #
        ###########
        fig, axes = plt.subplots(1,3)
        # [axi.set_axis_off() for axi in axes.ravel()]

        # Plot Predicted Variance Against Observed/Empirical Variance
        # x = [overall_match_var[m][r]['pred_below'] for r in ranges]
        # y = [overall_match_var[m][r]['obs_below'] for r in ranges]
        x = np.array([overall_match_var[m][r]['pred'].cpu().numpy() for r in ranges])
        y = np.array([overall_match_var[m][r]['obs'].cpu().numpy() for r in ranges])

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
        x = np.array([overall_match_var[m][r]['pred'].cpu().numpy() for r in ranges])
        y = np.array([overall_match_var[m][r]['obs'].cpu().numpy() for r in ranges])

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

        path = "{}/{}/{}".format(logdir,'calibration',m)
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
