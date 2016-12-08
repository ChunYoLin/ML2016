import sys
import re
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
import matplotlib.pyplot as plt
from preprocess import Corpus
tf.python.control_flow_ops = tf

title = Corpus('title')

fea = title.bow * np.log(len(title.corpus) / title.df)

input_wordvec = Input(shape = (fea.shape[1], ))
encoded = Dense(2000, activation = 'relu')(input_wordvec)
encoded = Dense(1000, activation = 'relu')(encoded)
encoded = Dense(500, activation = 'relu')(encoded)
code = Dense(5, activation = 'linear')(encoded)
encoded = Dense(5, activation = 'relu')(encoded)
decoded = Dense(500, activation = 'relu')(encoded)
decoded = Dense(1000, activation = 'relu')(decoded)
decoded = Dense(2000, activation = 'relu')(decoded)
decoded = Dense(fea.shape[1], activation = 'relu')(decoded)
autoencoder = Model(input = input_wordvec, output = decoded)
autoencoder.compile(optimizer = 'adam', loss = 'mse')

autoencoder.fit(fea, fea,
                nb_epoch = 10,
                batch_size = 200,
                shuffle = True,
                )
encoder = Model(input = input_wordvec, output = code)
code = encoder.predict(fea)
Group = KMeans(n_clusters = 21, random_state = 0).fit_predict(code)
tag = np.zeros(shape = 21, dtype = np.int32)
for c in Group:
    tag[c] += 1
print len(title.word_set)
print tag
print code[0:10]
reduced_data = TSNE(n_components = 2).fit_transform(code)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group * 5, s = 20)
plt.show()
with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')

