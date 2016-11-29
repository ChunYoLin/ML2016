from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import OrderedDict
import numpy as np
import re
import draw_cluster
import matplotlib.pyplot as plt
import nltk

corpus = []
title = []
word_set = {}
with open('./title_StackOverflow.txt') as f1, open('./docs.txt') as f2:
    for row in f1:
        corpus.append(re.split('\W+', row.lower()))
        title.append(re.split('\W+', row.lower()))
    for row in f2:
        corpus.append(re.split('\W+', row.lower()))
for doc in corpus:
    df_list = []
    for word in doc:
        word = word.lower()
        if word not in word_set:
            word_set[word] = 1
            df_list.append(word)
        elif word not in df_list:
            word_set[word] += 1
            df_list.append(word)
word_set_sorted = OrderedDict(sorted(word_set.items(), key = lambda x: x[1], reverse = True))

ignore_word = set()
stopwords = nltk.corpus.stopwords.words('english')
for w in stopwords:
    ignore_word.add(w)
IDX = 0
for k, v in word_set_sorted.iteritems():
    IDX += 1
    if IDX <= 20:
        print k
        ignore_word.add(k)
    if v < 4:
        ignore_word.add(k)

fea_num = 100

model = Word2Vec(size = fea_num, window = 8, min_count = 4, workers = 8)
model.build_vocab(corpus)
for i in range(4):
    model.train(corpus)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
print model.similar_by_word('windows')
wd_class = 250
wd_vec = []
wd_id = {}
wid = 0
for wv in model.vocab:
    if wv not in ignore_word:
        wd_vec.append(model[wv])
        wd_id[wv] = wid
        wid += 1
word_group = KMeans(n_clusters = wd_class, random_state = 0, max_iter = 1000).fit_predict(wd_vec)
df = np.zeros(shape = wd_class)
print 'start cal df'
for doc in corpus:
    df_list = []
    for word in doc:
        if word not in df_list and word not in ignore_word:
            df[word_group[wd_id[word]]] += 1
            df_list.append(word)
print 'finish cal df'
for x in df:
    print x
title_bow = np.zeros(shape = (len(title), wd_class))
for i in range(len(title)):
    for w in title[i]:
        if w not in ignore_word:
            title_bow[i, word_group[wd_id[w]]] += 1
title_bow *= np.log(len(corpus) / df)
center = []
center_num = 20
center_arg = []
for i in range(center_num):
    if i == 0:
        center.append(title_bow[0])
        center_arg.append(0)
    else:
        inner = np.inner(center[i - 1], title_bow)
        for xid, x in enumerate(np.argsort(np.abs(inner))):
            print 'search %d, inner product %g'%(xid, inner[x])
            if x not in center_arg and np.sum(title_bow[x]) != 0:
                print 'choose %d, inner product %g'%(xid, inner[x])
                center.append(title_bow[x])
                center_arg.append(x)
                break
center = np.asarray(center)
Group = KMeans(n_clusters = 20, init = 'k-means++', random_state = 0, max_iter = 1000).fit_predict(title_bow)
Group = KMeans(n_clusters = 20, init = center, random_state = 0, max_iter = 1000).fit_predict(title_bow)

tag = np.zeros(shape = 20, dtype = np.int32)
for c in Group:
    tag[c] += 1
print tag
with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')
reduced_data = PCA(n_components = 2).fit_transform(title_bow)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group + 1, s = 20)
plt.show()
