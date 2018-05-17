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

    def get_negative_samples(self, vocab_size):
        return torch.LongTensor([int(i) for i in 
            torch.Tensor(NEGATIVE_SAMPLE_SIZE).random_(to=vocab_size)
        ])

    def forward(self, x, y):
        x_embeddings = self.word_to_embedding(x)
        batch_size, emb_size = x_embeddings.size()
        x_embeddings = x_embeddings.view(batch_size, 1, emb_size)
        y_embeddings = self.embedding_to_context(y).view(batch_size, emb_size, 1)

        positive_output = F.logsigmoid(
            x_embeddings @ 
            y_embeddings
        ).view(-1, 1)

        vocab_size = self.embedding_to_context.weight.size()[0]
        negative_samples = self.get_negative_samples(vocab_size)
        negative_embeddings = self.embedding_to_context(negative_samples).neg()
        negative_output = F.logsigmoid(
            negative_embeddings @
             x_embeddings.view(batch_size, emb_size, 1)
        ).sum(1)
        return - (positive_output + negative_output).mean()

    def most_similar(self, index, n_samples):
        vector = self.word_to_embedding.weight[index, :]
        cosine_scores = cosine_similarity(vector.detach().reshape(1, -1),
            self.word_to_embedding.weight.detach().numpy())

        indices = list(reversed(cosine_scores.argsort()[0]))[2:n_samples+1]
        return indices, list(cosine_scores[0, indices])
