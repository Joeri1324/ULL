import csv
import numpy as np


def write_predictions(testfile, model, vocab):
    with open(testfile) as infile:
        with open('predictions.txt', 'w') as outfile:
            predictions = csv.writer(outfile, delimiter='\t')
            test = csv.reader(infile, delimiter='\t')
            for target, sentence_id, position, sentence in test:
                # context = get_context(sentence, int(position))
                candidates, scores = model.predict(target)
                line = ['\t' + target + ' ' + str(sentence_id)]
                for i in candidates:
                    line.append(vocab[i] + ' '  + str(scores[i]))
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
    # test_scores = [0.7, 0.65, 0.62, 0.9]
    # test_words = ['table', 'chair', 'bike', 'kitchen']
    write_predictions('lst/lst_test.preprocessed', None)