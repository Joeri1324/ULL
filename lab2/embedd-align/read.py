def read(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        for line1, line2 in zip(f1, f2):
            yield line1, line2

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