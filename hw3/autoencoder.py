import cPickle as pk
import numpy as np
import tensorflow as tf

#---network parameter---#
n_input = 3076
n_hidden_1 = 256
n_hidden_2 = 128

X = tf.placeholder(shape = (None, n_input))
weighs = {
        'encoder_h1': tf.Variable(tf.random_normal(shape = (n_input, n_hidden_1))),
        'encoder_h2': tf.Variable(tf.random_normal(shape = (n_hidden_1, n_hidden_2))),
        'decoder_h1': tf.Variable(tf.random_normal(shape = (n_hidden_2, n_hidden_1))),
        'decoder_h2': tf.Variable(tf.random_normal(shape = (n_hidden_1, n_input))),
}
biases = {
        'encoder_b1': tf.Variable(tf.random_normal(shape = (n_hidden_1)))},
        'encoder_b2': tf.Variable(tf.random_normal(shape = (n_hidden_2)))},
        'decoder_b1': tf.Variable(tf.random_normal(shape = (n_hidden_1)))},
        'decoder_b2': tf.Variable(tf.random_normal(shape = (n_input)))},
}
#---building encoder and decoder---#
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weighs['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weighs['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weighs['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weighs['decoder_h2']), biases['decoder_b2']))
    return layer_2
#  construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
#  prediction
y_pred = decoder_op
y_true = X
#  define loss
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(2e-4).minimize(cost)

init = tf.initialize_all_variables()

with tf.session as sess:
    sess.run(init)

