import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

NEGATIVE_SAMPLE_SIZE = 20

class Embeddings(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Embeddings, self).__init__()
        self.word_to_embedding = nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.embedding_to_context = nn.Embedding(vocab_size, embedding_size, sparse=True)
        self.negative_sampler = torch.distributions.uniform.Uniform(torch.tensor([0]), torch.tensor([vocab_size]))

    def forward(self, x, y):
        # positive
        positive_output = - F.logsigmoid(
            self.word_to_embedding(x) @ self.embedding_to_context(y).view(-1, 1)
        )

        chicken = self.word_to_embedding(x) @ self.embedding_to_context(y).view(-1, 1)

        



        vocab_size = self.embedding_to_context.weight.size()[0]
        # should use word frequency for this
        negative_samples = torch.LongTensor([int(i) for i in 
            torch.Tensor(NEGATIVE_SAMPLE_SIZE).random_(to=vocab_size)
        ])
        fish = self.embedding_to_context(negative_samples).neg() @  self.word_to_embedding(x).view(-1, 1)


        negative_output = - F.logsigmoid(
            self.embedding_to_context(negative_samples).neg() @  self.word_to_embedding(x).view(-1, 1)
        )
        return (positive_output + negative_output.sum())

    def most_similar(self, index):
        vector = self.word_to_embedding.weight[:, index]
        most_similar = cosine_similarity(vector.detach().reshape(1, -1),
            self.word_to_embedding.weight.detach().numpy().T).argsort()

        return most_similar[0, -2]
