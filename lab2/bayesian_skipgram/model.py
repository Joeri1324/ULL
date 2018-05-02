import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


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

        self.negative_sampler = torch.distributions.uniform.Uniform(torch.tensor([0]), torch.tensor([vocab_size]))

    def forward(self, x, context):

        Rw = self.M(self.Embedding(x).repeat(len(context), x.size()[0]))
        Rc = self.M(self.Embedding(context))

        h = F.relu(torch.cat((Rw, Rc), 1)).sum(0)
        mu = self.U(h)
        sigma = F.softplus(self.W(h))

        prior = dist.multivariate_normal.MultivariateNormal(
            self.prior_mus(x),
            covariance_matrix=torch.diagflat(torch.mul(self.prior_sigmas(x), self.prior_sigmas(x)))
        )
        posterior = dist.multivariate_normal.MultivariateNormal(
            mu,
            covariance_matrix=torch.diagflat(sigma)
        )
        vocab_size = self.Embedding.weight.size()[0]

        likelihood = 0
        for j in context:
            positive = dist.multivariate_normal.MultivariateNormal(
                self.prior_mus(j),
                covariance_matrix=torch.diagflat(torch.mul(self.prior_sigmas(j), self.prior_sigmas(j)))
            )
            negative_samples = torch.LongTensor([int(i) for i in 
                torch.Tensor(NEGATIVE_SAMPLE_SIZE).random_(to=vocab_size)
            ])
            for k in negative_samples:

                negative = dist.multivariate_normal.MultivariateNormal(
                    self.prior_mus(k),
                    covariance_matrix=torch.diagflat(torch.mul(self.prior_sigmas(k), self.prior_sigmas(k)))
                )
                likelihood += max(0, dist.kl.kl_divergence(posterior, negative)
                                  - dist.kl.kl_divergence(posterior, positive) + 1) 
                                  # 1 should b m for batches
        return likelihood - dist.kl.kl_divergence(posterior, prior)
