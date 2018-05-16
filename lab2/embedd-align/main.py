from read import read, read_vocab
from model import AlignedEmbeddings
import torch.optim as optim
import torch

FILE_L1 = './training.en'
FILE_L2 = './training.fr'

vocab_l1 = {w: i for i, w in enumerate(read_vocab(FILE_L1, []))}
vocab_l2 = {w: i for i, w in enumerate(read_vocab(FILE_L2, []))}

model = AlignedEmbeddings(12, len(vocab_l1), len(vocab_l2), 10)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train():
    losses = []
    for l1, l2 in read(FILE_L1, FILE_L2):
        optimizer.zero_grad()
        model.zero_grad()
        l1_tensor = torch.LongTensor([vocab_l1[w] for w in l1.split()])
        l2_tensor = torch.LongTensor([vocab_l2[w] for w in l2.split()])

        loss = model(l1_tensor, l2_tensor)
        loss.backward()
        losses.append(loss)
        optimizer.step()
    print('loss', torch.mean(torch.Tensor(losses)))

train()