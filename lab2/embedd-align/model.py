import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.multivariate_normal as normal
import torch.distributions.kl as kl

NEGATIVE_SAMPLE_SIZE = 10

class AlignedEmbeddings(nn.Module):

    def __init__(self, embedding_size, vocab_size_l1, vocab_size_l2, hidden_size):
        super(AlignedEmbeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size_l1, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True)
        self.U = nn.Linear(hidden_size*2, hidden_size)
        self.S = nn.Linear(hidden_size*2, hidden_size)
        self.f = nn.Linear(hidden_size, vocab_size_l1)
        self.g = nn.Linear(hidden_size, vocab_size_l2)

    def forward(self, words_l1, words_l2):
        embeddings = self.embeddings(words_l1)
        h, _ = self.lstm(embeddings.view(len(words_l1), 1, -1))
        u = self.U(h)
        s = F.softplus(self.S(h)) # have to check dim

        def get_kl(u1, s1, u2, s2):
            covariance_1 = torch.diagflat(s1 * s1)
            dist_1 = normal.MultivariateNormal(u1, covariance_matrix=covariance_1)

            covariance_2 = torch.diagflat(s2 * s2)
            dist_2 = normal.MultivariateNormal(u2, covariance_matrix=covariance_2)

            return kl.kl_divergence(dist_1, dist_2)

        dim = u.size()[2]
        I = torch.ones(dim)
        zeros = torch.zeros(dim)

        kl_loss = sum(get_kl(mu, sigma, zeros, I) for mu, sigma in zip(u, s))

        loss = - kl_loss
        for mu, sigma, word in zip(u, s, words_l1):
            ### here do the log P(xi|zi)
            covariance = torch.diagflat(sigma * sigma)
            dist = normal.MultivariateNormal(mu, covariance_matrix=covariance)

            z = dist.sample()
            loss += F.log_softmax(self.f(z), dim=1)[0, word]

        for word in words_l2:
            for mu, sigma in zip(u, s):
                covariance = torch.diagflat(sigma * sigma)
                dist = normal.MultivariateNormal(mu, covariance_matrix=covariance)

                z = dist.sample()
                loss += 1 / len(words_l2) * F.log_softmax(self.g(z), dim=1)[0, word] 

        return loss