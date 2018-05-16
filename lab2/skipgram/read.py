from itertools import filterfalse
import torch

def read_sequential(file_name, context_window_size, filters, vocab, batch_size=32):
    with open(file_name) as file:
        X = []
        Y = []
        for line in file:
            splitted_line = line.split()
            sentence_length = len(splitted_line)
            for i, word in enumerate(splitted_line):
                context_window_indices = filterfalse(
                    lambda x: x == i,
                    range(max(0, i - context_window_size), 
                          min(sentence_length, i + context_window_size + 1)),
                )

                for j in context_window_indices:
                    context_word = splitted_line[j]
                    if (not any(f(context_word) for f in filters) 
                        and not any(f(word) for f in filters)):
                        X.append(word)
                        Y.append(context_word)
                        if len(X) == batch_size:
                            yield torch.LongTensor([vocab[w] for w in X]), torch.LongTensor([vocab[w] for w in Y])
                            X = []
                            Y = []

# def batch_read(sequential_reader, vocab, batch_size=32):
#     for x, y in sequential_reader:
        
#     x, y = zip(*[sequential_reader.__next__() for _ in range(batch_size)])
#     yield torch.LongTensor([vocab[w] for w in x]), torch.LongTensor([vocab[w] for w in y])
    

def read_context_wise(file_name, context_window_size, filters):
    with open(file_name) as file:
        for line in file:
            splitted_line = line.split()
            sentence_length = len(splitted_line)
            for i, word in enumerate(splitted_line):
                context_window_indices = filterfalse(
                    lambda x: x == i,
                    range(max(0, i - context_window_size), 
                            min(sentence_length, i + context_window_size + 1)),
                )

                # didn't do the filters yet
                yield word, [splitted_line[j] for j in context_window_indices]

def read_vocab(file_name, filters):
    seen = {}
    with open(file_name) as file:
        for line in file:
            for word in line.split():
                if (not any(f(context_word) for f in filters) 
                    and not any(f(word) for f in filters) 
                    and not seen.get(word, None)):
                    seen[word] = True
                    yield word

def destructure(iterator):
    return [x for x in iterator]
