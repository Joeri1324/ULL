from itertools import filterfalse

def read(file_name, context_window_size, filters):
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

                for j in context_window_indices:
                    context_word = splitted_line[j]
                    if (not any(f(context_word) for f in filters) 
                        and not any(f(word) for f in filters)):
                        yield word, context_word

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
