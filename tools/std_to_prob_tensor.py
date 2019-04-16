import numpy as np
import matplotlib.pyplot as plt
import torch
import time

num_images = 20
num_classes = 11 #11

mus = torch.normal(torch.zeros(num_images,num_classes,512,512),torch.ones(num_images,num_classes,512,512))
sigmas = torch.normal(torch.ones(num_images,num_classes,512,512),0.1*torch.ones(num_images,num_classes,512,512))
G = torch.distributions.normal.Normal(0,1)

print(mus.shape)
print(mus.repeat(1,2,1,1).shape)


exit()

print((mus>sigmas))
ms = torch.cat((mus.unsqueeze(0),sigmas.unsqueeze(0)),0)

print((mus>sigmas).shape)
print(ms.shape)

x = ms.gather(0,(mus>sigmas).long().unsqueeze(0)).squeeze(0)
print(x.shape)
exit()

# probability that the largest number is greater than all others
t = time.time()
max_mu_i = torch.argmax(mus,dim=1)
max_mu = mus.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)
max_sigma = sigmas.gather(1,max_mu_i.clone().unsqueeze(1)).squeeze(1)

cum_cdf = torch.zeros(max_mu_i.shape)
for x in np.arange(-1,1,0.1):
    x_scale = max_mu + 5*max_sigma*x

    cdf = torch.ones(cum_cdf.shape)    
    
    for i in range(num_classes):
        cdf = torch.mul( cdf, G.cdf((x_scale-mus[:,i,:,:])/sigmas[:,i,:,:]) )
    cdf = cdf + torch.exp(G.log_prob((x_scale-max_mu)/(max_sigma)))

    cum_cdf = cum_cdf + cdf

