import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.multivariate_normal as normal
import torch.distributions.kl as kl
from itertools import product

NEGATIVE_SAMPLE_SIZE = 10

class BayesianSkipgram(nn.Module):
    
    def __init__(self, embedding_size, vocab_size, context_size):
        super(BayesianSkipgram, self).__init__()

        # Have to check the dimension embedding size
        self.Embedding =  nn.Embedding(vocab_size, embedding_size)
        self.M = nn.Linear(embedding_size, context_size)
        self.U = nn.Linear(context_size*2, context_size)
        self.W = nn.Linear(context_size*2, context_size)

        # have to think of proper initialization
        self.prior_mus = nn.Embedding(vocab_size, context_size)
        self.prior_sigmas = nn.Embedding(vocab_size, context_size)

    def most_similar(self, index):
        p_covariance = torch.diagflat(self.prior_sigmas(index) * self.prior_sigmas(index))
        p_index = normal.MultivariateNormal(self.prior_mus(index), covariance_matrix=p_covariance)
        vocab_size = self.Embedding.weight.size()[0]

        def calc(p_index, i):
            i = torch.LongTensor([i])
            covariance = torch.diagflat(self.prior_sigmas(i) * self.prior_sigmas(i))
            p = normal.MultivariateNormal(self.prior_mus(i), covariance_matrix=covariance)
            return (i, kl.kl_divergence(p_index, p))

        return min([calc(p_index, i) for i in range(vocab_size)], key=lambda x: x[1])

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

        def compute_likelihood(j, k):
            j_sigma = self.prior_sigmas(j)
            j_covariance = torch.diagflat(j_sigma * j_sigma)
            positive = normal.MultivariateNormal(self.prior_mus(j), covariance_matrix=j_covariance)
            k_sigma = self.prior_sigmas(k)
            k_covariance = torch.diagflat(k_sigma * k_sigma)
            negative = normal.MultivariateNormal(self.prior_mus(k), covariance_matrix=k_covariance)
            return max(0,  kl.kl_divergence(posterior, positive) - kl.kl_divergence(posterior, negative) + 1) 
            
        likelihood = sum(compute_likelihood(j, k) for j, k in product(context, negative_samples))

        return  kl.kl_divergence(posterior, prior) + likelihood, kl.kl_divergence(posterior, prior)
