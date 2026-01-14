import torch 
import numpy as np
import matplotlib.pyplot as plt


class VELoss:
    """ The variance exploding loss function for training score-based generative models.
        Args:
            marginal_prob_std: A function that gives the standard deviation of 
              the perturbation kernel.
            eps: A tolerance value for numerical stability.

    """

    def __init__(self, marginal_prob_std, eps=1e-5):
        self.marginal_prob_std = marginal_prob_std
        self.eps = eps

    def __call__(self, net, x, month, srf):
        """ Args:
            net: A PyTorch network instance that represents a 
              time-dependent score-based model.
            x: A mini-batch of training data.    
            month: A mini-batch of corresponding month conditions.
        """
        #print("__call__(self, net, x, month, srf, ts)")
        #print(x.shape, month.shape, srf.shape, ts.shape)
        random_t = torch.rand(x.shape[0], device=x.device) * (1. - self.eps) + self.eps  
        z = torch.randn_like(x)
        std = self.marginal_prob_std(random_t)

        perturbed_x = x + z * std[:, None, None, None]
        score = net(perturbed_x, random_t, month, srf)

        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))

        return loss

