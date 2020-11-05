import numpy as np
import matplotlib.pyplot as plt
import torch
import time

num_classes = 11 #11

# create one gaussian per class
G = []
mus = []
sigmas = []

for i in range(num_classes):

    # gaussian parameters generated programmatically
    mu = torch.normal(torch.tensor([0],dtype=torch.double),torch.tensor([1.0],dtype=torch.double))
    sigma = torch.normal(torch.tensor([1],dtype=torch.double),torch.tensor([0.1],dtype=torch.double))
    
    # create gaussian
    dist = torch.distributions.normal.Normal(torch.tensor([mu],dtype=torch.double),torch.tensor([sigma],dtype=torch.double))

    G.append(dist)
    mus.append(mu.numpy()[0])
    sigmas.append(sigma.numpy()[0])

mus = np.array(mus)
sigmas = np.array(sigmas)

plt.figure()
for g in G:
    x = torch.from_numpy(np.arange(-5,5,0.01))
    y = torch.exp(g.log_prob(x))

    plt.plot(x.numpy(),y.numpy())


# print means and stds
for i in range(len(G)):
    print("{}: {} {}".format(i,mus[i],sigmas[i]))



# probability that the largest number is greater than all others
t = time.time()
max_mu_i = np.argmax(mus)
max_mu = mus[max_mu_i]
max_sigma = sigmas[max_mu_i]

cum_cdf = 0
lower = max_mu-5*max_sigma
upper = max_mu+5*max_sigma
step = max_sigma/1
for x in np.arange(lower,upper,step):
    cdf = []
    for i,g in enumerate(G):
        if i!=max_mu_i:
            cdf.append( g.cdf(x) )
    cdf.append( torch.exp(G[max_mu_i].log_prob(x)) )
    cum_cdf += np.prod(cdf)*step
print(time.time()-t)

# what is the probability that samples from the largest mean Gaussian are actually the largest?

# sampling proof of concept
t = time.time()

n_samples = 100000
samples = torch.zeros(num_classes,n_samples)
for i,g in enumerate(G):
    samples[i,:] = g.sample(torch.Size([n_samples]))[:,0]

print(torch.argmax(samples,dim=0))
print(torch.argmax(samples,dim=0)==max_mu_i)

largest_num = torch.sum(torch.argmax(samples,dim=0)==max_mu_i).numpy()
total_num = n_samples

cum_cdf_numeric = 1.*largest_num/total_num

print(time.time()-t)

print("Analytic: {}".format(cum_cdf))
print("Numeric: {}".format(cum_cdf_numeric))

# for step_i in [1,10,100,1000]:
#     # probability that the largest number is greater than all others
#     max_mu_i = np.argmax(mus)
#     max_mu = mus[max_mu_i]
#     max_sigma = sigmas[max_mu_i]

#     cum_cdf = 0
#     lower = max_mu-5*max_sigma
#     upper = max_mu+5*max_sigma
#     step = 1.*max_sigma/step_i
#     for x in np.arange(lower,upper,step):
#         cdf = []
#         for i,g in enumerate(G):
#             if i!=max_mu_i:
#                 cdf.append( g.cdf(x) )
#         cdf.append( torch.exp(G[max_mu_i].log_prob(x)) )
#         cum_cdf += np.prod(cdf)*step

#     print("Analytic {}: {}".format(step_i,cum_cdf))


plt.show()
