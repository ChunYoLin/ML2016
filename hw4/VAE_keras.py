import sys
import re
import numpy as np
import tensorflow as tf
from keras.layers import Input, LSTM, RepeatVector, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
import matplotlib.pyplot as plt
from preprocess import Corpus
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

tf.python.control_flow_ops = tf
title = Corpus('title')
fea = title.bow * np.log(len(title.corpus) / title.df)
batch_size = 100
original_dim = fea.shape[1]
latent_dim = 100
intermediate_dim = 256
nb_epoch = 10
epsilon_std = 1.0
x = Input(shape = (fea.shape[1], ))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape = (batch_size, latent_dim), mean=0., std = epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon
z = Lambda(sampling, output_shape = (latent_dim,))([z_mean, z_log_sigma])
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

x_train = fea

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        )
code = encoder.predict(fea)
Group = KMeans(n_clusters = 20, random_state = 0).fit_predict(code)
tag = np.zeros(shape = 20, dtype = np.int32)
for c in Group:
    tag[c] += 1
print len(title.word_set)
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

reduced_data = PCA(n_components = 2).fit_transform(code)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group * 5, s = 20)
plt.show()
