import torch.nn as nn

from ptsemseg.models.utils import *

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.isotonic import IsotonicRegression as IR
from sklearn.linear_model import LogisticRegression as LR


class HistogramLinearRecalibrator():
    def __init__(self, c, ranges, device):
        self.c = c
        self.ranges = ranges
        self.device = device

        self.W = torch.ones(1, device=device)
        self.b = torch.zeros(1, device=device)

    def fit(self, output, label):
        confidence, accuracy = calcClassStatistics(output, label, self.ranges, self.c)
        self.W = torch.ones(len(confidence), dtype=torch.float, device=self.device)
        self.b = torch.zeros(len(confidence), dtype=torch.float, device=self.device)

        confidence, accuracy = zip(*sorted(zip(confidence, accuracy)))
        confidence = list(confidence)
        accuracy = list(accuracy)

        XX = zip(confidence[:-1], confidence[1:])
        YY = zip(accuracy[:-1], accuracy[1:])

        for i, XY in enumerate(zip(XX, YY)):
            X, Y = XY
            x1, x2 = X
            y1, y2 = Y

            self.W[i] = 1. * (y2 - y1) / (x2 - x1)
            self.b[i] = y2 - self.W[i] * x2

    def predict(self, x):
        self.W = self.W.to(x.device)
        self.b = self.b.to(x.device)

        i = (1. * len(self.W) * torch.clamp(x, min=0, max=1)).floor().long() - 1

        return self.W[i] * x + self.b[i]


class HistogramFlatRecalibrator():
    def __init__(self, c, ranges, device):
        self.c = c
        self.ranges = ranges
        self.device = device

        self.b = torch.zeros(1, device=device)

    def fit(self, output, label):
        confidence, accuracy = calcClassStatistics(output, label, self.ranges, self.c)
        self.b = torch.zeros(len(confidence), device=self.device, dtype=torch.float)

        confidence = confidence.to(self.device)
        accuracy = accuracy.to(self.device)

        confidence, accuracy = zip(*sorted(zip(confidence, accuracy)))
        confidence = list(confidence)
        accuracy = list(accuracy)

        XX = zip(confidence[:-1], confidence[1:])
        YY = zip(accuracy[:-1], accuracy[1:])

        for i, XY in enumerate(zip(XX, YY)):
            X, Y = XY
            x1, x2 = X
            y1, y2 = Y

            self.b[i] = 0.5 * (y1 + y2)

    def predict(self, x):
        self.b.to(x.device)

        i = (1. * len(self.b) * torch.clamp(x, min=0, max=1)).floor().long() - 1

        return self.b[i]


class PolynomialRecalibrator():
    def __init__(self, c, ranges, degree, device):
        self.c = c
        self.ranges = ranges
        self.device = device
        self.model = PolynomialRegressionModel(degree, device)

        self.criterion = nn.MSELoss()
        self.l_rate = 0.01
        self.optimiser = torch.optim.SGD(self.model.parameters(), lr=self.l_rate)
        self.epochs = 2000
        self.device = device

    def fit(self, output, label):
        confidence, accuracy = calcClassStatistics(output, label, self.ranges, self.c)
        for epoch in range(self.epochs):
            self.optimiser.zero_grad()
            y_pred = self.model.forward(confidence)
            loss = self.criterion(y_pred, accuracy)
            loss.backward()
            self.optimiser.step()
            print("Epoch {}, Loss {}".format(epoch, loss.data.cpu().numpy()))

    def predict(self, x):
        with torch.no_grad():
            out = self.model.forward(x)
        return out


class IsotonicRecalibrator():
    def __init__(self, c, device):
        self.c = c
        self.ir = IR(out_of_bounds = 'clip')
        self.device = device

    def fit(self, output, label):
        x = output[:,self.c,:,:].reshape(-1).data.cpu().numpy().astype(np.float)
        y = (label == self.c).reshape(-1).data.cpu().numpy().astype(np.float)
        self.ir.fit(x, y)

    def predict(self, x):
        shape = x.shape
        x = x.reshape(-1).data.cpu().numpy().astype(np.float)

        return torch.tensor(self.ir.transform(x), device=self.device, dtype=torch.float).reshape(shape)


class PlattRecalibrator():  # logistic regression
    def __init__(self, c, device):
        self.c = c
        self.lr = LR(solver="lbfgs")
        self.device = device

    def fit(self, output, label):
        x = output[:,self.c,:,:].reshape(-1, 1).data.cpu().numpy().astype(np.float)
        y = (label == self.c).reshape(-1).data.cpu().numpy().astype(np.float)

        self.lr.fit(x, y)

    def predict(self, x):
        shape = x.shape
        x = x.reshape(-1).data.cpu().numpy().astype(np.float)
        return torch.tensor(self.lr.predict_proba(x.reshape(-1, 1))[:, 1], device=self.device, dtype=torch.float).reshape(shape)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, degree, device):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.W = torch.nn.Parameter(data=torch.ones(self.degree, 1, device=device, dtype=torch.float),
                                    requires_grad=True)

    def forward(self, x):
        self.W = self.W.to(x.device)
        X = torch.cat(tuple([torch.pow(x, i).unsqueeze(1) for i in range(self.degree)]), 1)
        return torch.matmul(X, self.W).squeeze()


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


def calcClassStatistics(output, label, ranges, c):

    pred = c
    gt = label
    pred_var = output[:,c,:,:]

    confidences = []
    accuracies = []

    for r in ranges:

        low, high = r
        idx_pred_gt_match = (pred == gt)  # everywhere correctly labeled to correct class
        idx_pred_var_in_range = (low <= pred_var) & (pred_var < high)  # everywhere with specified confidence level

        sumval_pred_var_in_range = torch.sum(pred_var[idx_pred_var_in_range], dtype=torch.float)
        num_obs_var_in_range = torch.sum((idx_pred_gt_match & idx_pred_var_in_range), dtype=torch.float)

        num_in_range = torch.sum(idx_pred_var_in_range, dtype=torch.float)

        total = num_in_range if num_in_range > 0 else 1.0
        confidence = sumval_pred_var_in_range / total
        accuracy = num_obs_var_in_range / total

        if num_in_range == 0:
            confidence = torch.tensor((low + high) / 2.0, device=output.device, dtype=torch.float)

        confidences.append(confidence)
        accuracies.append(accuracy)

    confidences = torch.stack(confidences)
    accuracies = torch.stack(accuracies)
    torch.cuda.empty_cache()

    return confidences, accuracies


def calcStatistics(output, label, ranges):

    pred_var, pred = torch.max(output, dim=1)
    gt = label

    confidences = []
    accuracies = []

    for r in ranges:
        low, high = r

        idx_pred_gt_match = (pred == gt)  # everywhere correctly labeled to correct class
        idx_pred_var_in_range = (low <= pred_var) & (pred_var < high)  # everywhere with specified confidence level

        sumval_pred_var_in_range = torch.sum(pred_var[idx_pred_var_in_range], dtype=torch.float)
        num_obs_var_in_range = torch.sum((idx_pred_gt_match & idx_pred_var_in_range), dtype=torch.float)

        num_in_range = torch.sum(idx_pred_var_in_range, dtype=torch.float)

        total = num_in_range if num_in_range > 0 else 1.0
        confidence = sumval_pred_var_in_range/(total)
        accuracy = num_obs_var_in_range/(total)

        if num_in_range == 0:
            confidence = torch.tensor((low + high) / 2.0, device=output.device, dtype=torch.float)

        confidences.append(confidence)
        accuracies.append(accuracy)

    confidences = torch.stack(confidences)
    accuracies = torch.stack(accuracies)
    torch.cuda.empty_cache()

    return confidences, accuracies
