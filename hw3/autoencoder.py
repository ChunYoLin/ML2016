import numpy as np
import tensorflow as tf
import input_data
import scipy.spatial.distance
import cPickle as pk
from sklearn.neighbors import NearestNeighbors

n_input = 3072
f_num = [3, 128, 128, 32, 64]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
#  encode_layer
W_en_conv1 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[0], f_num[1]], stddev = 0.01))
b_en_conv1 = tf.Variable(tf.constant(value = 0., shape = [f_num[1]]))
W_en_conv2 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[1], f_num[2]], stddev = 0.01))
b_en_conv2 = tf.Variable(tf.constant(value = 0., shape = [f_num[2]]))
W_en_conv3 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[2], f_num[3]], stddev = 0.01))
b_en_conv3 = tf.Variable(tf.constant(value = 0., shape = [f_num[3]]))
W_en_conv4 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[3], f_num[4]], stddev = 0.01))
b_en_conv4 = tf.Variable(tf.constant(value = 0., shape = [f_num[4]]))
#  decode layer
W_de_conv1 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[4], f_num[3]], stddev = 0.01))
b_de_conv1 = tf.Variable(tf.constant(value = 0., shape = [f_num[3]]))
W_de_conv2 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[3], f_num[2]], stddev = 0.01))
b_de_conv2 = tf.Variable(tf.constant(value = 0., shape = [f_num[2]]))
W_de_conv3 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[2], f_num[1]], stddev = 0.01))
b_de_conv3 = tf.Variable(tf.constant(value = 0., shape = [f_num[1]]))
W_de_conv4 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[1], f_num[0]], stddev = 0.01))
b_de_conv4 = tf.Variable(tf.constant(value = 0., shape = [f_num[0]]))

#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#---model definition---#
sess = tf.InteractiveSession(config = config)

#---building encoder and decoder---#
def encoder(l, x):
    if l >= 1:
        x = tf.reshape(x, [-1, 32, 32, 3])
        h_conv1 = conv2d(x, W_en_conv1) + b_en_conv1
        #  h_norm1 = tf.nn.lrn(h_conv1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a1 = tf.nn.relu(h_conv1)
        h_pool1 = tf.nn.max_pool(h_a1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        x = h_pool1
    if l >= 2:
        h_conv2 = conv2d(x, W_en_conv2) + b_en_conv2
        #  h_norm2 = tf.nn.lrn(h_conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool(h_a2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        x = h_pool2
    if l >= 3:
        h_conv3 = conv2d(x, W_en_conv3) + b_en_conv3
        #  h_norm3 = tf.nn.lrn(h_conv3, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a3 = tf.nn.relu(h_conv3)
        h_pool3 = tf.nn.max_pool(h_a3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        x = h_pool3
    if l >= 4:
        h_conv4 = conv2d(x, W_en_conv4) + b_en_conv4
        #  h_norm4 = tf.nn.lrn(h_conv4, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a4 = tf.nn.relu(h_conv4)
        h_pool4 = tf.nn.max_pool(h_a4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        x = h_pool4
    return x

def decoder(l, x):
    if l >= 4: 
        h_conv1 = conv2d(x, W_de_conv1) + b_de_conv1
        #  h_norm1 = tf.nn.lrn(h_conv1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a1 = tf.nn.relu(h_conv1)
        b, h, w, c = h_a1.get_shape().as_list()
        h_upsamp1 = tf.image.resize_images(h_a1, (h * 2, w * 2))
        x = h_upsamp1
    if l >= 3:    
        h_conv2 = conv2d(x, W_de_conv2) + b_de_conv2
        #  h_norm2 = tf.nn.lrn(h_conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a2 = tf.nn.relu(h_conv2)
        b, h, w, c = h_a2.get_shape().as_list()
        h_upsamp2 = tf.image.resize_images(h_a2, (h * 2, w * 2))
        x = h_upsamp2
    if l >= 2:
        h_conv3 = conv2d(x, W_de_conv3) + b_de_conv3
        #  h_norm3 = tf.nn.lrn(h_conv3, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a3 = tf.nn.relu(h_conv3)
        b, h, w, c = h_a3.get_shape().as_list()
        h_upsamp3 = tf.image.resize_images(h_a3, (h * 2, w * 2))
        x = h_upsamp3
    if l >= 1:
        h_conv4 = conv2d(x, W_de_conv4) + b_de_conv4
        #  h_norm4 = tf.nn.lrn(h_conv4, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
        h_a4 = tf.nn.relu(h_conv4)
        b, h, w, c = h_a4.get_shape().as_list()
        h_upsamp4 = tf.image.resize_images(h_conv4, (h * 2, w * 2))
        x = h_upsamp4
    return x

#---network parameter---#
X = tf.placeholder(tf.float32, shape = (None, n_input))
y_ = tf.placeholder(tf.float32, shape = (None, 10))
keep_prob = tf.placeholder(tf.float32)
sess.run(tf.initialize_all_variables())
#  preprocessing data
dataset = input_data.CIFAR10()
labeled_image, label = dataset.labeled_image()
unlabeled_image = dataset.unlabeled_image()
all_image = np.concatenate((labeled_image, unlabeled_image), axis = 0) / 255.
batch_size = 50
batch = input_data.minibatch(all_image, batch_size = batch_size)
labeled_image /= 255.
unlabeled_image /= 255.
for l in range(1, 4, 1):
#---construct model---#
    encoder_op = encoder(l, X)
    #  autoencoder model
    decoder_op = tf.reshape(decoder(l, encoder_op), [-1, n_input])
    y_pred = decoder_op
    y_true = X
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    #  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_true))
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)
    init_new_vars_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_new_vars_op)
    #---train the autoencoder---#
    e = 10
    for epoch in range(e):
        for i in range(batch.shape[0]):
            _, c, y_p, y_t = sess.run([optimizer, cost, y_pred, y_true], feed_dict = {X: all_image[batch[i]]})
            print 'fine tune, num of layer', l
            print 'epoch ' + str(epoch + 1) + '/'+ str(e) + ' batch '  + str(i + 1) + '/' + str(batch.shape[0]) +' cost:' + str(c) 

        
#---encode label and unlabel image---#

#  label_code = sess.run(encoder_op, feed_dict = {X: labeled_image})
#---validate---#
train_image = []
train_label = []
validate_image = []
validate_label = []
train_set_size = 400
for i in range(10):
    for j in range(500):
        if j < train_set_size:
            train_image.append(labeled_image[i * 500 + j])
            train_label.append(label[i * 500 + j])
        else:
            validate_image.append(labeled_image[i * 500 + j])
            validate_label.append(label[i * 500 + j])
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
validate_image = np.asarray(validate_image)
validate_label = np.asarray(validate_label)
train_code = []
for i in range(40):
    train_code.append(sess.run(encoder_op, feed_dict = {X: train_image[i * 100 : (i + 1) * 100]}))
train_code = np.asarray(train_code).reshape(4000, -1)
validate_code = sess.run(encoder_op, feed_dict = {X: validate_image})
label_code = []
for i in range(0, train_set_size * 10, train_set_size):
    label_code.append(np.mean(train_code[i : i + train_set_size], axis = 0))
label_code = np.asarray(label_code)
validate_code = validate_code.reshape(1000, -1)
self_label = np.ndarray(shape = (validate_code.shape[0], 10))
#  simularity = np.argmin(scipy.spatial.distance.cdist(validate_code, train_code, 'euclidean'), axis = 1)
simularity = scipy.spatial.distance.cdist(validate_code, label_code, 'cosine')
orig_im = []
for i in range(0, train_set_size * 10, train_set_size):
    orig_im.append(np.mean(train_image[i : i + train_set_size], axis = 0))
orig_im = np.asarray(orig_im)
orig_SIM = scipy.spatial.distance.cdist(validate_image, orig_im, 'cosine')
for K in [1, 3, 5, 10, 20, 50, 100]:
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
    print 'knn accuracy', acc / 1000
    print '-------------------'
acc = np.zeros(10)
correct = np.zeros(10)
incorrect = np.zeros(10)
for i in range(1000):
    if np.argmax(validate_label[i]) == np.argmin(orig_SIM[i]):
        acc[np.argmax(validate_label[i])] += 1.
        correct[np.argmax(validate_label[i])] += np.min(orig_SIM[i])
        #  print "image %d, correct, label %d, code_label %d, code distance %g"%(i, np.argmax(validate_label[i]), np.argmin(simularity[i]), np.min(simularity[i]))
    else:
        incorrect[np.argmax(validate_label[i])] += np.min(orig_SIM[i])
print 'avg correct dist', correct / acc
print 'avg incorrect dist', incorrect / (100 - acc)
print 'accuracy', acc / 100.
print 'overall accuracy', np.mean(acc) / 100.

print '----------------------------------'

acc = np.zeros(10)
correct = np.zeros(10)
incorrect = np.zeros(10)
for i in range(1000):
    if np.argmax(validate_label[i]) == np.argmin(simularity[i]):
        acc[np.argmax(validate_label[i])] += 1.
        correct[np.argmax(validate_label[i])] += np.min(simularity[i])
        #  print "image %d, correct, label %d, code_label %d, code distance %g"%(i, np.argmax(validate_label[i]), np.argmin(simularity[i]), np.min(simularity[i]))
    else:
        incorrect[np.argmax(validate_label[i])] += np.min(simularity[i])
        #  print "image %d, incorrect, label %d, code_label %d, code distance %g"%(i, np.argmax(validate_label[i]), np.argmin(simularity[i]), np.min(simularity[i]))
print 'avg correct dist', correct / acc
print 'avg incorrect dist', incorrect / (100 - acc)
print 'accuracy', acc / 100.
print 'overall accuracy', np.mean(acc) / 100.

#  acc = 0.
#  for i in range(self_label.shape[0]):
    #  self_label[i] = train_label[simularity[i]]
    #  if np.argmax(self_label[i]) == np.argmax(validate_label[i]):
        #  acc += 1.
#  print acc / self_label.shape[0]

#  unlabel_code = []
#  for i in range(100):
    #  code = sess.run(encoder_op, feed_dict = {X: unlabeled_image[i * 450 : (i + 1) * 450]})
    #  unlabel_code.append(code)

#  unlabel_code = np.asarray(unlabel_code).reshape(45000, -1)
#  label_code = label_code.reshape(5000, -1)
#  print label_code.shape, unlabel_code.shape

#  #---label unlabel_image by encode_code cosine simiularity---#
#  self_label = np.ndarray(shape = (unlabel_code.shape[0], 10))
#  unlabel_simularity = np.argmin(scipy.spatial.distance.cdist(unlabel_code, label_code, 'cosine'), axis = 1)
#  simularity = scipy.spatial.distance.cdist(unlabel_code, label_code, 'cosine')
#  for i in range(self_label.shape[0]):
    #  print simularity[i]
    #  self_label[i] = label[unlabel_simularity[i]]

#  labeled_image = np.concatenate((labeled_image, unlabeled_image), axis = 0)
#  label = np.concatenate((label, self_label), axis = 0)
#  print 'dump the image and label......'

#  with open('image.pk', 'w') as f:
    #  pk.dump(labeled_image, f)
#  with open('label.pk', 'w') as f:
    #  pk.dump(label, f)


