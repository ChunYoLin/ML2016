import numpy as np
import input_data
import scipy.spatial.distance
import cPickle as pk
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
n_input = 3072
f_num = [3, 32, 32, 64, 64, 64]
#  encode_layer
W_en_conv1 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[0], f_num[1]], stddev = 0.01))
b_en_conv1 = tf.Variable(tf.constant(value = 0., shape = [f_num[1]]))
W_en_conv2 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[1], f_num[2]], stddev = 0.01))
b_en_conv2 = tf.Variable(tf.constant(value = 0., shape = [f_num[2]]))
W_en_conv3 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[2], f_num[3]], stddev = 0.01))
b_en_conv3 = tf.Variable(tf.constant(value = 0., shape = [f_num[3]]))
W_en_conv4 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[3], f_num[4]], stddev = 0.01))
b_en_conv4 = tf.Variable(tf.constant(value = 0., shape = [f_num[4]]))
W_en_conv5 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[4], f_num[5]], stddev = 0.01))
b_en_conv5 = tf.Variable(tf.constant(value = 0., shape = [f_num[5]]))

l = 5
X = tf.placeholder(tf.float32, shape = (None, n_input))
X_image = tf.reshape(X, [-1, 32, 32, 3])
x = X_image 
if l >= 1:
    h_conv1 = conv2d(X_image, W_en_conv1) + b_en_conv1
    h_a1 = tf.nn.relu(h_conv1)
    x = h_a1
if l >= 2:
    h_conv2 = conv2d(x, W_en_conv2) + b_en_conv2
    h_a2 = tf.nn.relu(h_conv2)
    h_pool2 = tf.nn.max_pool(h_a2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    x = h_pool2
if l >= 3:
    h_conv3 = conv2d(x, W_en_conv3) + b_en_conv3
    h_a3 = tf.nn.relu(h_conv3)
    h_pool3 = tf.nn.max_pool(h_a3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    x = h_pool3
if l >= 4:
    h_conv4 = conv2d(x, W_en_conv4) + b_en_conv4
    h_a4 = tf.nn.relu(h_conv4)
    h_pool4 = tf.nn.max_pool(h_a4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    x = h_pool4
if l >= 5:
    h_conv5 = conv2d(x, W_en_conv5) + b_en_conv5
    h_a5 = tf.nn.relu(h_conv5)
    h_pool5 = tf.nn.max_pool(h_a5, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    x = h_pool5

#  preprocessing data
dataset = input_data.CIFAR10()
labeled_image, label, train_image, train_label, validate_image, validate_label = dataset.labeled_image()
train_image /= 255.
validate_image /= 255.
labeled_image /= 255.
train_code = []
CNN_validate_label = pk.load(open('./validate_l', 'r'))
#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)
saver = tf.train.Saver()
saver.restore(sess, "./autoencoder_model")

for i in range(40):
    train_code.append(sess.run(x, feed_dict = {X: train_image[i * 100 : (i + 1) * 100]}))
train_code = np.asarray(train_code).reshape(4000, -1)
validate_code = []
for i in range(10):
    validate_code.append(sess.run(x, feed_dict = {X: validate_image[i * 100 : (i + 1) * 100]}))
validate_code = np.asarray(validate_code).reshape(1000, -1)
validate_code = validate_code.reshape(1000, -1)
print 'dump the code..'
with open('train_code', 'w') as f:
    pk.dump(train_code, f)
with open('validate_code', 'w') as f:
    pk.dump(validate_code, f)

for K in [1, 3, 5, 10, 20, 30, 50, 100, 1000]:
    nbrs = NearestNeighbors(n_neighbors = K, algorithm = 'auto', p = 2, n_jobs = 4).fit(train_code)
    distances, indices = nbrs.kneighbors(validate_code)
    acc = 0.
    for i in range(1000):
        label_count = np.zeros(shape = (1000, 10))
        for j in range(K):
            neigh_cls = np.argmax(train_label[indices[i, j]])
            label_count[i, neigh_cls] += np.exp(-1. * distances[i, j])
        if np.argmax(label_count[i]) == np.argmax(validate_label[i]):
            acc += 1
    print 'K value:', K
    print 'code knn accuracy', acc / 1000
    print '-------------------'
    nbrs = NearestNeighbors(n_neighbors = K, algorithm = 'auto', p = 2, n_jobs = 4).fit(train_image)
    distances, indices = nbrs.kneighbors(validate_image)
    acc = 0.
    for i in range(1000):
        label_count = np.zeros(shape = (1000, 10))
        for j in range(K):
            neigh_cls = np.argmax(train_label[indices[i, j]])
            label_count[i, neigh_cls] += np.exp(-1 * distances[i, j])
        if np.argmax(label_count[i]) == np.argmax(validate_label[i]):
            acc += 1
    print 'K value:', K
    print 'raw data knn accuracy', acc / 1000
    print '-------------------'
    print '=============================='

