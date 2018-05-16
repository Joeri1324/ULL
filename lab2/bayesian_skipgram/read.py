from itertools import filterfalse

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
                c = [splitted_line[j] for j in context_window_indices]
                if len(c) > 0:
                    yield word, c

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