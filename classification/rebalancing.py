import torch

def prior_recbalancing(logit,beta,s_prior,t_prior=None):
	# logit (b,c,h,w): pre-softmax network output
	# beta (1,): user controlled hyperparameter 
	# s_prior (1,c): source (training) data prior
	# t_prior (1,c): target (test) data prior (most likely uniform)

    prob = torch.nn.Softmax(dim=1)(logit) 

    inv_prior = 1/s_prior
    inv_prior[inv_prior == float("inf")] = 0
    inv_prior = inv_prior.unsqueeze(0).float()

    if t_prior is None:
        prob_r = prob*inv_prior
    else:
        prob_r = prob*inv_prior*t_prior

    prob_r = prob_r/prob_r.sum(1).unsqueeze(1) # nomalize to make valid prob
    
    outputs = prob**(1-beta) * prob_r**beta
    outputs = outputs/outputs.sum(1).unsqueeze(1) # nomalize to make valid prob
    return outputs