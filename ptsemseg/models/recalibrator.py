import torch.nn as nn

from ptsemseg.models.utils import *

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class HistogramLinearRecalibrator():
    def __init__(self,c,ranges,device):

        self.c = c
        self.ranges = ranges
        self.device = device

        self.W = torch.ones(1,device=device)
        self.b = torch.zeros(1,device=device)

    def fit(self,confidence,accuracy):

        self.W = torch.ones(len(confidence),device=self.device)
        self.b = torch.zeros(len(confidence),device=self.device)

        W = [0]*len(confidence)
        b = [0]*len(confidence)

        confidence, accuracy = zip(*sorted(zip(confidence,accuracy)))
        confidence = list(confidence)
        accuracy = list(accuracy)

        XX = zip(confidence[:-1],confidence[1:])
        YY = zip(accuracy[:-1],accuracy[1:])

        for i,XY in enumerate(zip(XX,YY)):
            X,Y = XY
            x1,x2 = X
            y1,y2 = Y
            
            self.W[i] = 1.*(y2-y1)/(x2-x1)
            self.b[i] = y2-self.W[i]*x2


        self.W.to(self.device)
        self.b.to(self.device)


    def predict(self,x):

        self.W.to(x.device)
        self.b.to(x.device)

        i = (1.*len(self.W)*torch.clamp(x,min=0,max=1)).floor().long()-1

        return self.W[i]*x + self.b[i]



class HistogramFlatRecalibrator():
    def __init__(self,c,ranges,device):

        self.c = c
        self.ranges = ranges
        self.device = device

        self.b = torch.zeros(1,device=device)

    def fit(self,output,label):

        confidence, accuracy = calcStatistics(self,output,label)



        self.b = torch.zeros(len(confidence),device=self.device)

        confidence, accuracy = zip(*sorted(zip(confidence,accuracy)))
        confidence = list(confidence)
        accuracy = list(accuracy)

        XX = zip(confidence[:-1],confidence[1:])
        YY = zip(accuracy[:-1],accuracy[1:])

        for i,XY in enumerate(zip(XX,YY)):
            X,Y = XY
            x1,x2 = X
            y1,y2 = Y
            
            self.b[i] = 0.5*(y1+y2)

        self.b.to(self.device)



    def predict(self,x):

        self.b.to(x.device)

        i = (1.*len(self.b)*torch.clamp(x,min=0,max=1)).floor().long()-1

        return self.b[i]
    

class PolynomialRecalibrator():
    def __init__(self,c, ranges, degree,device):

        self.c = c
        self.ranges = ranges
        self.device = device
        self.model = PolynomialRegressionModel(degree,device)

        self.criterion = nn.MSELoss()
        self.l_rate = 0.01
        self.optimiser = torch.optim.SGD(self.model.parameters(),lr=self.l_rate)
        self.epochs = 2000
        self.device = device

    def fit(self,confidence,accuracy):

        for epoch in range(self.epochs):
            self.optimiser.zero_grad()
            y_pred = self.model.forward(confidence)
            loss = self.criterion(y_pred,accuracy)
            loss.backward()
            self.optimiser.step()
            print("Epoch {}, Loss {}".format(epoch,loss.data.cpu().numpy()))


    def predict(self,x):
        with torch.no_grad():
            out = self.model.forward(x)
        return out

class IsotonicRecalibrator():
    def __init__(self,device):
        self.device = device
        self.ir = IsotonicRegression()

    def fit(self,confidence,accuracy,device):

        x = confidence.data.cpu().numpy()
        y = accuracy.data.cpu().numpy()
        self.ir.fit(x, y)

    def predict(self,x):
    
        x = confidence.data.cpu().numpy()
        
        return torch.tensor(self.ir.predict(x))
  

class PlattRecalibrator(): # i.e. logistic regression
    def __init__(self,device):
        self.device = device
        self.lr = LogisticRegression()               

    def fit(self,confidence,accuracy,device):

        x = confidence.data.cpu().numpy()
        y = accuracy.data.cpu().numpy()
                           
        self.lr.fit(x, y)
        
    def predict(self,x):
    
        x = confidence.data.cpu().numpy()
        
        return torch.tensor(self.lr.predict_proba(x.reshape(-1, 1))[:,1])

# TODO Temperature scaling recalibration : https://arxiv.org/pdf/1706.04599.pdf

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__() 
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, degree, device):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.W = torch.nn.Parameter(data=torch.ones(self.degree,1,device=device,dtype=torch.float), requires_grad=True)        
    def forward(self, x):
        self.W = self.W.to(x.device)
        X = torch.cat(tuple([torch.pow(x,i).unsqueeze(1) for i in range(self.degree)]),1)
        return torch.matmul(X,self.W).squeeze()

# class CalibrationNet(torch.nn.Module):
#     # def __init__(self, n_feature, n_hidden, n_output):
#     def __init__(self):
#         super(CalibrationNet, self).__init__()
#         n_feature = 1
#         n_hidden = 100
#         n_output = 1
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x



def calcStatistics(self,mean,variance,label,ranges,c):
    softmax_mu = mean #torch.nn.Softmax(1)(mean[m])
    variance = softmax_mu

    max_mu_i = torch.argmax(mean,dim=1)
    max_mu = mean.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)
    max_sigma = softmax_mu.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)

    # Define Ground Truth, Prediction, and Associated Prediction Probability
    pred = max_mu_i
    gt = label
    pred_var = max_sigma #softmax_mu
    
    for r in ranges:
        pred = c
        pred_var = softmax_mu[:,c,:,:]

        low,high = r
        idx_pred_gt_match = (pred==gt) #&(pred==c) # everywhere correctly labeled to correct class
        idx_pred_var_in_range = (low<=pred_var)&(pred_var<high) # everywhere with specified confidence level

        sumval_pred_var_in_range = torch.sum(pred_var[idx_pred_var_in_range])
        num_obs_var_in_range = torch.sum((idx_pred_gt_match&idx_pred_var_in_range))

        num_in_range = torch.sum(idx_pred_var_in_range)
        num_correct = torch.sum(idx_pred_gt_match) 

        total = total if total>0 else 1
        confidence = 1.*per_class_match_var[m][r][c]['sumval_pred_in_range']/total 
        accuracy = 1.*per_class_match_var[m][r][c]['num_obs_in_range']/total 

        if per_class_match_var[m][r][c]['num_in_range']==0:
            confidence  = (low+high)/2.0

        return confidence, accuracy

            
def fitCalibration(outputs,labels,ranges,n_classes,m,device,cfg):

    # TODO test and debug isotonic and platt recalibration
    if recal == "Isotonic" or recal == "Platt":
        x = outputs[m]
        y = labels
    else:
        x = np.array([per_class_match_var[m][r][c]['pred'] for r in ranges])
        y = np.array([per_class_match_var[m][r][c]['obs'] for r in ranges])

        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

# Fit Calibration if Not Already
    calibrationPerClass[m][c]['model'].fit(x,y,device)
    calibrationPerClass[m][c]['fit'] = True

    return calibrationPerClass

def showCalibration(calibrationPerClass, ranges,m,logdir,n_classes,device):

    ###########
    # Overall #
    ###########
    fig, axes = plt.subplots(1,3)
    plt.tight_layout()

    # Plot Predicted Variance Against Observed/Empirical Variance
    x = np.array([overall_match_var[m][r]['pred'] for r in ranges])
    y = np.array([overall_match_var[m][r]['obs'] for r in ranges])

    xp = np.arange(0,1,0.001)
    yp = np.interp(xp,x,y)



    # TODO fix plotting with invalid probilities and graph wrapping
    axes[0].plot(x,y,'.')
    axes[0].set_title("Uncalibrated")
    axes[1].set_xlabel("softmax score")
    axes[1].set_ylabel("emperical probability")


    # Convert Predicted Variances to Calibrated Variances
    x = np.array([overall_match_var[m][r]['pred'] for r in ranges])
    y = np.array([overall_match_var[m][r]['obs'] for r in ranges])

    # x = torch.from_numpy(x.reshape(-1,1)).float()
    x = torch.from_numpy(x).float()

    y_pred = calibration[m]["model"].predict(x.to(device))
    y_pred = y_pred.cpu().numpy()

    print(y, y_pred)

    # y_pred = calibration[m].predict(x)
    # y_pred = calibration[m].predict(x[:,np.newaxis])
    axes[1].plot(y_pred,y)                    
    axes[1].set_title("Recalibrated")
    axes[1].set_xlabel("calibrated confidence")
    axes[1].set_ylabel("emperical probability")

    # Recalibration Curve
    x = np.arange(0,1,0.001)
    x = torch.from_numpy(x).float()

    # TODO figure out why we are calibrating the already recalibrated class scores?
    y = calibration[m]["model"].predict(x.to(device))

    x = x.cpu().numpy()
    y = y.cpu().numpy()

    # y = calibration[m].predict(x)
    # y = calibration[m].predict(x[:,np.newaxis])
    axes[2].plot(x,y)                    
    axes[2].set_title("Recalibration Curve")
    axes[2].set_xlabel("softmax probability")
    axes[2].set_ylabel("calibrated confidence")


    # calculating expected calibration error
    #ECE = np.sum(np.absolute(y - y_pred))
    #fig.suptitle('Expected Calibration Error: {}'.format(ECE), fontsize=16)

    path = "{}/{}/{}".format(logdir,'calibration',m)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/calibratedOverall{}.png".format(path,i))



    plt.close(fig)



    ############################
    # All Classes Uncalibrated #
    ############################
    fig, axes = plt.subplots(3,n_classes//3+1)
    # [axi.set_axis_off() for axi in axes.ravel()]

    for c in range(n_classes):
        # x = [per_class_match_var[m][r][c]['pred_below'] for r in ranges]
        # y = [per_class_match_var[m][r][c]['obs_below'] for r in ranges]                        
        x = [per_class_match_var[m][r][c]['pred'] for r in ranges]
        y = [per_class_match_var[m][r][c]['obs'] for r in ranges]                                        
        axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].plot(x,y)
        axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].set_title("Class: {}".format(c))

    path = "{}/{}/{}".format(logdir,'calibration',m)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/uncalibratedPerClass{}.png".format(path,i))
    plt.close(fig)

    ##########################
    # All Classes Calibrated #
    ##########################
    fig, axes = plt.subplots(3,n_classes//3+1)
    # [axi.set_axis_off() for axi in axes.ravel()]

    for c in range(n_classes):
        # x = [per_class_match_var[m][r][c]['pred_below'] for r in ranges]
        # y = [per_class_match_var[m][r][c]['obs_below'] for r in ranges]                        
        x = np.array([per_class_match_var[m][r][c]['pred'] for r in ranges])
        y = np.array([per_class_match_var[m][r][c]['obs'] for r in ranges])

        # x = torch.from_numpy(x.reshape(-1,1)).float()
        x = torch.from_numpy(x).float()

        y_pred = calibrationPerClass[m][c]["model"].predict(x.to(device))
        y_pred = y_pred.cpu().numpy()

        axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].plot(y,y_pred)
        axes[(c+1)//(n_classes//3+1),(c+1)%(n_classes//3+1)].set_title("Class: {}".format(c))

    path = "{}/{}/{}".format(logdir,'calibration',m)
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig("{}/calibratedPerClass{}.png".format(path,i))
    plt.close(fig)

    # print(overall_match_var[m])    
