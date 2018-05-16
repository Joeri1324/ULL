import torch
from read import read_context_wise, read_vocab
from model import BayesianSkipgram
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch

FILE = '../data/training.en'

vocab = {w: i for i, w in enumerate(read_vocab(FILE, []))}
inv_vocab = {i: w for i, w in enumerate(read_vocab(FILE, []))}

model = BayesianSkipgram(50, len(vocab), 4)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train():
    losses = []
    fishes = []
    for word, context in read_context_wise(FILE, 2, []):
        word_index = torch.LongTensor([vocab[word]])
        optimizer.zero_grad()
        model.zero_grad()

        
        context_indices = torch.LongTensor([vocab[i] for i in context])
        print(word, context)
        loss, fish = model(word_index, context_indices)
        losses.append(loss)
        fishes.append(fish)
        loss.backward()
        optimizer.step()
    print('loss', torch.mean(torch.Tensor(losses)), 'fish', torch.mean(torch.Tensor(fishes)))

for _ in range(100):
    train()