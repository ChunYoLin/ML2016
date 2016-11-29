from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.cluster import KMeans
import numpy as np
import re
import draw_cluster
class LabeledLineSentence(object):
        def __init__(self, filename):
            self.filename = filename
        def __iter__(self):
            for uid, line in enumerate(open(self.filename)):
                yield LabeledSentence(words = line.lower().split(), tags = [u'SENT_%s' % uid])
            #  x = uid
            #  for uid, line in enumerate(open('./docs.txt')):
                #  yield LabeledSentence(words = line.lower().split(), tags = [u'SENT_%s' % (uid + x)])
f = "./reduce_title.txt"
sentences = LabeledLineSentence(f)
doc = []
fea_num = 100
with open(f) as F:
    for idx, row in enumerate(F):
        doc.append(row.strip('\n'))
model = Doc2Vec(size = fea_num, window = 8, min_count = 4, workers = 8)
model.build_vocab(sentences)
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
title = np.zeros(shape = (20000, fea_num))
for i in range(20000):
    title[i] = model.docvecs[i]
Group = KMeans(n_clusters = 20, random_state = 0).fit_predict(title)
f_o = [open('./class_out/class%d'%(k), 'w') for k in range(20)]
for i in range(20000):
    for j in range(20):
        if Group[i] == j:
            f_o[j].write(doc[i] + '\n')
            
    
tag_word = ['wordpress', 'oracle', 'svn', 'apache', 'excel', 'matlab', 'visual-studio',
        'cocoa', 'osx', 'bash', 'spring', 'hibernate', 'scala', 'sharepoint', 'ajax', 'qt', 'drupal',
        'linq', 'haskell', 'magento']
tag = np.zeros(shape = 20, dtype = np.int32)
for c in Group:
    tag[c] += 1
print tag
#  draw_cluster.plot(title)

with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')

