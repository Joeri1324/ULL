from read import read_sequential, destructure, read_vocab
from model import Embeddings
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch

EMBEDDINGS_SIZE = 100
FILE = 'training.en'

vocab = {w: i for i, w in enumerate(read_vocab(FILE, []))}
model = Embeddings(EMBEDDINGS_SIZE, len(vocab))

loss_function = nn.NLLLoss()
optimizer = optim.SparseAdam(model.parameters(), lr = 0.001)

def train():
    losses = []
    i = 0
    for x, y in read_sequential(FILE, 2, [], vocab, 500):
        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        i += 1
    return torch.mean(torch.tensor(losses))

for _ in range(20):
    print('Loss:', train())

with open('pickled_model_emb_size_{0}'.format(EMBEDDINGS_SIZE), 'wb') as f:
    model.save(f)
