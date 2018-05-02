import torch
from read import read_context_wise, read_vocab
from model import BayesianSkipgram

FILE = '../data/dev.en'

vocab = {w: i for i, w in enumerate(read_vocab(FILE, []))}
inv_vocab = {i: w for i, w in enumerate(read_vocab(FILE, []))}

model = BayesianSkipgram(10, len(vocab), 4)

for word, context in read_context_wise(FILE, 2, []):
    word_index = torch.LongTensor([vocab[word]])

    context_indices = torch.LongTensor([vocab[i] for i in context])
    model(word_index, context_indices)

    break