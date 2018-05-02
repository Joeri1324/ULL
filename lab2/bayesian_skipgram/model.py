import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class BayesianSkipgram(nn.Module):
    
    def __init__(self, embedding_size, vocab_size, context_size):
        super(BayesianSkipgram, self).__init__()

        # Have to check the dimension embedding size
        self.Embedding =  nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.M = nn.Linear(embedding_size, context_size)
        self.U = nn.Linear(context_size*2, context_size)
        self.W = nn.Linear(context_size*2, context_size)

        # have to think of proper initialization
        self.prior_mus = nn.Embedding(vocab_size, context_size, sparse=True)
        self.prior_sigmas = nn.Embedding(vocab_size, context_size, sparse=True)

    def forward(self, x, context):
        Rw = self.M(self.Embedding(x).repeat(len(context), x.size()[0]))
        Rc = self.M(self.Embedding(context))

        h = F.relu(torch.cat((Rw, Rc), 1)).sum(0)
        mu = self.U(h)
        sigma = F.softplus(self.W(h))

        prior = dist.multivariate_normal.MultivariateNormal(
            self.prior_mus(x),
            covariance_matrix=torch.diagflat(self.prior_sigmas(x))
        )
        posterior = dist.multivariate_normal.MultivariateNormal(
            mu,
            covariance_matrix=torch.diagflat(sigma)
        )

        divergence = dist.kl.kl_divergence(prior, posterior)
        print('divv', divergence)
