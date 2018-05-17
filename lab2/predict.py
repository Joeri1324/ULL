import csv
import numpy as np


def write_predictions(testfile, model, vocab):
    with open(testfile) as infile:
        with open('lst/predictions.txt', 'w') as outfile:
            predictions = csv.writer(outfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_NONE, escapechar=":")
            test = csv.reader(infile, delimiter='\t')
            for target, sentence_id, position, sentence in test:
                # context = get_context(sentence, int(position))
                candidates, scores = model.predict(target.split('.')[0])
                line = ['RANKED' + '\t' + target + ' ' + str(sentence_id)]
                for i, c in enumerate(candidates):
                    line.append(vocab[c] + ' '  + str(scores[i]))
                predictions.writerow(line)
                
           
def get_context(sentence, position, window=4):
    s = sentence.split(' ')
    context = []
    context_range = range(max(0,position-window), min(len(s), (position + window+1)))
    for w in context_range:
        if w == position: ## target word
            continue
        context.append(s[w])
    return context


if __name__ == '__main__':
    # vocab = {1:'kitchen', 2:'chair', 3:'bike', 4:'table'}
    # candidates = [3, 2, 1, 4]
    # scores = [0.8, 0.6, 0.4, 0.33]
    write_predictions('lst/lst_test.preprocessed', skipgram, vocab)