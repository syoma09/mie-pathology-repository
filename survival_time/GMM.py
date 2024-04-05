#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


minibatch = 32
num_gaussians = 10
in_features = 512
out_features = 512


class MDN(nn.Module):
    def __init__(self, in_features= in_features, out_features=out_features, num_gaussians=num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch=minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, mu, sigma


'''
    functions to compute the log-likelihood of data according to a gaussian mixture model.
    All the computations are done with log because exp will lead to numerical underflow errors.
'''
def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret =  torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(target, mu, sigma, pi):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)




##### Adding Noise ############

def add_noise(latent, noise_type="gaussian", sd=0.2):
    """Here we add noise to the latent features concatenated from the 4 autoencoders.
    Arguements:
    'gaussian' (string): Gaussian-distributed additive noise.
    'speckle' (string) : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    'sd' (float) : standard deviation used for generating noise

    Input :
        latent : numpy array or cuda tensor.

    Output:
        Array: Noise added input, can be np array or cuda tnesor.
    """
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.

        n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
        noise = n.sample(latent.size()).squeeze(2).cuda()
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn(latent.size()).cuda()
        latent = latent + latent * noise
        return latent



if __name__ == "__main__":
    model = MDN()
    model = model.cuda()
    print(model)