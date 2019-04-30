import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import os

class HistogramRecalibrator():
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




class LearnedRecalibrator():
    def __init__(self,degree,device):

        self.model = PolynomialRegressionModel(degree)

        self.criterion = nn.MSELoss()
        self.l_rate = 0.01
        self.optimiser = torch.optim.SGD(self.model.parameters(),lr=self.l_rate)
        self.epochs = 5000
        self.device = device

    def fit(self,x_init,y_init,device):

        self.device = device
        for epoch in range(self.epochs):
            self.optimiser.zero_grad()
            y_pred = self.model.forward(x_init)
            loss = self.criterion(y_pred,y_init)
            loss.backward()
            self.optimiser.step()
            print("Epoch {}, Loss {}".format(epoch,loss.data.cpu().numpy()))


    def predict(self,x):

        with torch.no_grad():
            out = self.model.forward(x)

        return out

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__() 
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, degree=2):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.W = torch.nn.Parameter(data=torch.ones(self.degree,1,dtype=torch.float), requires_grad=True)
        
    def forward(self, x):
        X = torch.cat(tuple([torch.pow(x,i).unsqueeze(1) for i in range((self.W.shape[0]))]),1)
        return torch.matmul(X,self.W).squeeze()



if __name__ == "__main__":
    print("HI")


    X = torch.from_numpy(np.arange(0,1,0.01)).float()
    Y = X + 0.1*torch.rand(X.shape[0]).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = LearnedRecalibrator(100,device)

    lr.fit(X,Y,device)

    Y_pred = lr.predict(X)



    X = X.numpy()
    Y = Y.numpy()
    Y_pred = Y_pred.numpy()

    plt.figure()
    plt.plot(X,Y)
    plt.plot(X,Y_pred)
    plt.show()