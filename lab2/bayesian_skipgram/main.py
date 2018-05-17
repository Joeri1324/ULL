import torch
from read import read_context_wise, read_vocab
from model import BayesianSkipgram
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch

FILE = '../data/dev.en'

vocab = {w: i for i, w in enumerate(read_vocab(FILE, []))}
inv_vocab = {i: w for i, w in enumerate(read_vocab(FILE, []))}

model = BayesianSkipgram(50, len(vocab))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train():
    losses = []
    for word, context in read_context_wise(FILE, 2, []):
        word_index = torch.LongTensor([vocab[word]])
        optimizer.zero_grad()
        model.zero_grad()
        context_indices = torch.LongTensor([vocab[i] for i in context])
        loss = model(word_index, context_indices)
        losses.append(loss)
        loss.backward()
        optimizer.step()
    print('loss', torch.mean(torch.Tensor(losses)))

for _ in range(50):
    train()
    indices, scores = model.most_similar(vocab['prostitute'])
    print(inv_vocab[indices[1]])
