import sys
import re
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from collections import OrderedDict
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import draw_cluster
import matplotlib.pyplot as plt
tf.python.control_flow_ops = tf

with open('./docs.txt', 'r') as f1, open('./title_StackOverflow.txt', 'r') as f2:
    word_set = {}
    doc = []
    doc_line = ''
    line = 0
    #  for row in f1:
        #  doc_line += row
        #  line += 1
        #  if line == 15:
            #  doc.append(doc_line)
            #  line = 0
            #  doc_line = ''
    title = []
    for row in f2:
        doc.append(row)
        title.append(row)
    for text in doc:
        idf_list = []
        for w in re.split('\W+', text):
            w = w.lower()
            if w not in word_set:
                word_set[w] = 1
                idf_list.append(w)
            elif w not in idf_list:
                word_set[w] += 1
                idf_list.append(w)

    word_set_sorted = OrderedDict(sorted(word_set.items(), key = lambda x: x[1], reverse = True))
    idx = 0
    del_set = set()
    for k, v in word_set_sorted.iteritems():
        idx += 1
        if idx <= 20:
            del_set.add(k)
            del word_set_sorted[k]
        if v <= 4:
            del_set.add(k)
            del word_set_sorted[k]
    word_set_id = {}
    w_id = 0
    for k in word_set_sorted.keys():
        word_set_id[k] = w_id
        w_id += 1
    doc_wd = np.zeros(shape = (len(doc), len(word_set_id)))
    title_wd = np.zeros(shape = (len(title), len(word_set_id)))
    for idx, d in enumerate(doc):
        for w in re.split('\W+', d):
            w = w.lower()
            if w not in del_set:
                doc_wd[idx, word_set_id[w]] += 1
    for idx, t in enumerate(title):
        for w in re.split('\W+', t):
            w = w.lower()
            if w not in del_set:
                title_wd[idx, word_set_id[w]] += 1

idf = np.zeros(shape = len(word_set_id))
for k, v in word_set_id.iteritems():
    idf[v] = word_set_sorted[k]
train_set = title_wd
idf = len(train_set) / idf
title_wd *= np.log(idf)
doc_wd *= np.log(idf)

input_wordvec = Input(shape = (len(word_set_id), ))
encoded = Dense(2000, activation = 'relu')(input_wordvec)
encoded = Dense(1000, activation = 'relu')(encoded)
encoded = Dense(500, activation = 'relu')(encoded)
code = Dense(5, activation = 'linear')(encoded)
encoded = Dense(5, activation = 'relu')(encoded)
decoded = Dense(500, activation = 'relu')(encoded)
decoded = Dense(1000, activation = 'relu')(decoded)
decoded = Dense(2000, activation = 'relu')(decoded)
decoded = Dense(len(word_set_id), activation = 'relu')(decoded)
autoencoder = Model(input = input_wordvec, output = decoded)
autoencoder.compile(optimizer = 'adam', loss = 'mse')

autoencoder.fit(train_set, train_set,
                nb_epoch = 10,
                batch_size = 200,
                shuffle = True,
                )
encoder = Model(input = input_wordvec, output = code)
code = encoder.predict(title_wd)
Group = KMeans(n_clusters = 20, random_state = 0).fit_predict(code)
tag = np.zeros(shape = 20, dtype = np.int32)
for c in Group:
    tag[c] += 1
print len(word_set_id)
print tag
print code[0:10]
with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')

reduced_data = PCA(n_components = 2).fit_transform(code)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group, s = 20)
plt.show()
