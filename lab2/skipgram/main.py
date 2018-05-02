from read import read, destructure, read_vocab
from model import Embeddings
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch

EMBEDDINGS_SIZE = 50
FILE = './data/test.en'

vocab = {w: i for i, w in enumerate(read_vocab(FILE, []))}
model = Embeddings(EMBEDDINGS_SIZE, len(vocab))

loss_function = nn.NLLLoss()
optimizer = optim.SparseAdam(model.parameters(), lr = 0.01)

def one_hot(index, size):
    vec = torch.zeros(size)
    vec[index] = 1
    return vec

size = len(destructure(read(FILE, 2, [])))
c = destructure(read_vocab(FILE, []))

def train():
    total_loss = 0
    i = 1
    losses = []
    for word, target in read(FILE, 2, []):
        
        i += 1
        optimizer.zero_grad()

        word_tensor = torch.LongTensor([vocab[word]])
        target_tensor = torch.LongTensor([vocab[target]])
        loss = model(word_tensor, target_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss)

    return torch.mean(torch.tensor(losses))

for _ in range(50):
    print('Loss:', train())



