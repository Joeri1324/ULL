import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.multivariate_normal as normal
import torch.distributions.kl as kl
import numpy as np
from itertools import product

NEGATIVE_SAMPLE_SIZE = 10

class BayesianSkipgram(nn.Module):
    
    def __init__(self, embedding_size, vocab_size):
        super(BayesianSkipgram, self).__init__()

        self.Embedding =  nn.Embedding(vocab_size, embedding_size)
        self.M = nn.Linear(embedding_size, embedding_size)
        self.U = nn.Linear(embedding_size*2, embedding_size)
        self.W = nn.Linear(embedding_size*2, embedding_size)

        self.prior_mus = nn.Embedding(vocab_size, embedding_size)
        self.prior_sigmas = nn.Embedding(vocab_size, embedding_size)

    def most_similar(self, index):
        prior_mus = self.prior_mus.weight.detach()
        prior_sigmas =  self.prior_sigmas.weight.detach()
        
        index = torch.LongTensor([index])
        distances = torch.abs(self.kl(
            self.prior_mus(index).view(-1), 
            self.prior_sigmas(index).view(-1), 
            prior_mus, prior_sigmas
        )).detach().numpy()

        indices = distances.argsort()
        return indices, distances[indices]

    def kl(self, mu0, cov0, mus1, covs1):
        var_1 = cov0 ** 2
        var_2 = covs1 ** 2

        mahala = 1 / (2 * var_1) * ((mu0 - mus1) ** 2 + var_1 - var_2)
        log = torch.log(torch.abs(covs1)) - torch.log(torch.abs(cov0))
        return (mahala + log).sum(1)

    def forward(self, x, context):

        Rw = self.M(self.Embedding(x).repeat(len(context), x.size()[0]))
        Rc = self.M(self.Embedding(context))

        h = F.relu(torch.cat((Rw, Rc), 1)).sum(0)
        mu = self.U(h)
        sigma = F.softplus(self.W(h))
        covariance = torch.diagflat(self.prior_sigmas(x) * self.prior_sigmas(x))

        prior = normal.MultivariateNormal(self.prior_mus(x), covariance_matrix=covariance)
        posterior = normal.MultivariateNormal(mu, covariance_matrix=torch.diagflat(sigma))
        vocab_size = self.Embedding.weight.size()[0]

        negative_samples = torch.LongTensor([int(i) for i in 
            torch.Tensor(NEGATIVE_SAMPLE_SIZE).random_(to=vocab_size)
        ])

        # can the abs be done in the embeddings datastructure?

        negative_kl = self.kl(mu, sigma, 
            torch.abs(self.prior_mus(negative_samples)), torch.abs(self.prior_sigmas(negative_samples)))

        positive_kl = self.kl(mu, sigma, 
            torch.abs(self.prior_mus(context)), torch.abs(self.prior_sigmas(context)))

        #print(negative_kl)
        m = 1
        likelihood = sum(max(0, x - y + m) for x, y in product(positive_kl, negative_kl))
        #print(likelihood)

        return  kl.kl_divergence(posterior, prior) + likelihood
