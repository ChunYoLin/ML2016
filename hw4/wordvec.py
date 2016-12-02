from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import re
import matplotlib.pyplot as plt
from preprocess import Corpus
import sys

corpus = Corpus(c = 'both', file_path = sys.argv[1])
title = Corpus(c = 'title', file_path = sys.argv[1])
fea_num = 100

#  model = Word2Vec(size = fea_num, window = 8, min_count = 4, workers = 8)
#  model.build_vocab(corpus.corpus)
#  for i in range(4):
    #  model.train(corpus.corpus)
    #  model.alpha -= 0.002
    #  model.min_alpha = model.alpha
#  print model.similar_by_word('windows')
#  model.save('./word2vec.model')
model = Word2Vec.load('./word2vec.model')

sent_vec = np.zeros(shape = (len(title.corpus), fea_num))
for i in range(len(title.corpus)):
    for w in title.corpus[i]:
        norm_len = 0.
        if w not in corpus.ig_word:
            norm_len += 1.
            weight = np.log((len(corpus.corpus) - corpus.word_set[w] + 0.5)/ (corpus.word_set[w] + 0.5))
            sent_vec[i] += model[w]
        if norm_len != 0:
            sent_vec[i] /= norm_len
Group = KMeans(n_clusters = 21, random_state = 0, max_iter = 1000).fit_predict(sent_vec)
tag = np.zeros(shape = 21, dtype = np.int32)
for c in Group:
    tag[c] += 1
print tag

with open(sys.argv[1] + 'check_index.csv', 'r') as f_in, open(sys.argv[2], 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])] and Group[int(p[1])] != np.argmax(tag):
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')
#  reduced_data = PCA(n_components = 2).fit_transform(sent_vec)
#  plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group * 5, s = 20)
#  plt.show()
