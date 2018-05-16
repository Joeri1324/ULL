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

    def get_negative_samples(self, vocab_size):
        return torch.LongTensor([int(i) for i in 
            torch.Tensor(NEGATIVE_SAMPLE_SIZE).random_(to=vocab_size)
        ])

    ## TODO: think of a way to make it a subset excluding the x's
    def forward(self, x, y):
        # positive
        x_embeddings = self.word_to_embedding(x)
        y_embeddings = self.embedding_to_context(y)
        batch_size, emb_size = x_embeddings.size()
        positive_output = - F.logsigmoid(torch.diag(x_embeddings @ y_embeddings.view(emb_size, batch_size)))

        vocab_size = self.embedding_to_context.weight.size()[0]
        negative_samples = self.get_negative_samples(vocab_size)
        negative_embeddings = self.embedding_to_context(negative_samples).neg()
        negative_output = - F.logsigmoid(negative_embeddings @  x_embeddings.view(emb_size, batch_size))
        return (positive_output.mean() + negative_output.sum())

    def most_similar(self, index):
        vector = self.word_to_embedding.weight[:, index]
        most_similar = cosine_similarity(vector.detach().reshape(1, -1),
            self.word_to_embedding.weight.detach().numpy().T).argsort()

        return most_similar[0, -2]
