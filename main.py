from model import load_model
from read import read_simlex
from scipy.stats import pearsonr
import numpy as np

model1 = load_model('deps.words')
model2 = load_model('bow2.words')
model3 = load_model('bow5.words')
simlex_bs = read_simlex('./SimLex-999/SimLex-999.txt')

model1_similaraties = [
    model1.similarity(s['word1'], s['word2'])
    for s in simlex_bs
    if s
]
model2_similaraties = [
    model2.similarity(s['word1'], s['word2'])
    for s in simlex_bs
    if s
]
model3_similaraties = [
    model3.similarity(s['word1'], s['word2'])
    for s in simlex_bs
    if s
]

print('PEARSON CORRELATION: MODEL 1',
      pearsonr(np.array(model1_similaraties),
      np.array([s['similarity'] for s in simlex_bs if s])))

print('PEARSON CORRELATION: MODEL 2',
      pearsonr(np.array(model2_similaraties),
      np.array([s['similarity'] for s in simlex_bs if s])))

print('PEARSON CORRELATION: MODEL 3',
      pearsonr(np.array(model3_similaraties),
      np.array([s['similarity'] for s in simlex_bs if s])))