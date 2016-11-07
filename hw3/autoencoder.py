import numpy as np
import tensorflow as tf
import input_data
import scipy.spatial.distance
import cPickle as pk

#---network parameter---#
n_input = 3072
n_hidden_1 = 512
#  n_hidden_2 = 256

X = tf.placeholder(tf.float32, shape = (None, n_input))
weighs = {
        'encoder_h1': tf.Variable(tf.random_normal(shape = (n_input, n_hidden_1))),
        #  'encoder_h2': tf.Variable(tf.random_normal(shape = (n_hidden_1, n_hidden_2))),
        #  'decoder_h1': tf.Variable(tf.random_normal(shape = (n_hidden_2, n_hidden_1))),
        'decoder_h2': tf.Variable(tf.random_normal(shape = (n_hidden_1, n_input))),
}
biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #  'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        #  'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
#---building encoder and decoder---#
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weighs['encoder_h1']), biases['encoder_b1']))
    #  layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weighs['encoder_h2']), biases['encoder_b2']))
    return layer_1

def decoder(x):
    #  layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weighs['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(x, weighs['decoder_h2']), biases['decoder_b2']))
    return layer_2

#---construct model---#
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
#  prediction
y_pred = decoder_op
y_true = X
#  define loss
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(2e-3).minimize(cost)
init = tf.initialize_all_variables()

#  preprocessing data
dataset = input_data.CIFAR10()
labeled_image, label = dataset.labeled_image()
unlabeled_image = dataset.unlabeled_image()
all_image = np.concatenate((labeled_image, unlabeled_image), axis = 0) / 255.
batch_size = 50
batch = input_data.minibatch(all_image, batch_size = batch_size)

#---run the graph---#
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        for i in range(batch.shape[0]):
            _, c, y_p, y_t = sess.run([optimizer, cost, y_pred, y_true], feed_dict = {X: all_image[batch[i]]})
        print 'epoch:' + str(epoch) + ' cost:' + str(c) 
        print 'y_pred:' + str(y_p) 
        print 'y_true:' + str(y_t)

#---encode label and unlabel image---#
    label_code = sess.run(encoder_op, feed_dict = {X: labeled_image})
    unlabel_code = sess.run(encoder_op, feed_dict = {X: unlabeled_image})

#---label unlabel_image by encode_code cosine simiularity---#
    self_label = np.ndarray(shape = (unlabel_code.shape[0], 10))
unlabel_simularity = np.argmin(scipy.spatial.distance.cdist(unlabel_code, label_code, 'euclidean'), axis = 1)
for i in range(self_label.shape[0]):
    self_label[i] = label[unlabel_simularity[i]]

labeled_image = np.concatenate((labeled_image, unlabeled_image), axis = 0)
label = np.concatenate((label, self_label), axis = 0)

with open('image.pk', 'w') as f:
    pk.dump(labeled_image, f)
with open('label.pk', 'w') as f:
    pk.dump(label, f)


