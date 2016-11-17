import numpy as np
import tensorflow as tf
import input_data
import scipy.spatial.distance
import cPickle as pk
from sklearn.neighbors import NearestNeighbors
import sys
n_input = 3072
L = 2
f_num = [3, 256, 512, 96, 192, 192, 192, 10]
f_size = [0, 3, 3, 3, 3, 3, 1, 1]
max_pool = [1, 2]
phase_train = tf.placeholder(tf.bool, name = 'phase_train')
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def conv2_2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 2, 2, 1], padding = 'SAME')
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
W_en_conv6 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[5], f_num[6]], stddev = 0.01))
b_en_conv6 = tf.Variable(tf.constant(value = 0., shape = [f_num[6]]))
W_en_conv7 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[6], f_num[7]], stddev = 0.01))
b_en_conv7 = tf.Variable(tf.constant(value = 0., shape = [f_num[7]]))
#  decode layer
W_de_conv7 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[7], f_num[6]], stddev = 0.01))
b_de_conv7 = tf.Variable(tf.constant(value = 0., shape = [f_num[6]]))
W_de_conv6 = tf.Variable(tf.random_normal(shape = [1, 1, f_num[6], f_num[5]], stddev = 0.01))
b_de_conv6 = tf.Variable(tf.constant(value = 0., shape = [f_num[5]]))
W_de_conv5 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[5], f_num[4]], stddev = 0.01))
b_de_conv5 = tf.Variable(tf.constant(value = 0., shape = [f_num[4]]))
W_de_conv4 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[4], f_num[3]], stddev = 0.01))
b_de_conv4 = tf.Variable(tf.constant(value = 0., shape = [f_num[3]]))
W_de_conv3 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[3], f_num[2]], stddev = 0.01))
b_de_conv3 = tf.Variable(tf.constant(value = 0., shape = [f_num[2]]))
W_de_conv2 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[2], f_num[1]], stddev = 0.01))
b_de_conv2 = tf.Variable(tf.constant(value = 0., shape = [f_num[1]]))
W_de_conv1 = tf.Variable(tf.random_normal(shape = [3, 3, f_num[1], f_num[0]], stddev = 0.01))
b_de_conv1 = tf.Variable(tf.constant(value = 0., shape = [f_num[0]]))


#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()
#---model definition---#
sess = tf.InteractiveSession(config = config)
#---building encoder and decoder---#
def encoder(l, x):
    if l >= 1:
        x = tf.reshape(x, [-1, 32, 32, 3])
        h_conv1 = conv2d(x, W_en_conv1) + b_en_conv1
        h_conv_bn_1 = input_data.batch_norm(h_conv1, f_num[1], phase_train)
        h_a1 = tf.nn.relu(h_conv_bn_1)
        h_pool1 = tf.nn.max_pool(h_a1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 1 in max_pool:
            x = h_pool1
        else:
            x = h_a1
    if l >= 2:
        h_conv2 = conv2d(x, W_en_conv2) + b_en_conv2
        h_conv_bn_2 = input_data.batch_norm(h_conv2, f_num[2], phase_train)
        h_a2 = tf.nn.relu(h_conv_bn_2)
        h_pool2 = tf.nn.max_pool(h_a2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 2 in max_pool:
            x = h_pool2
        else:
            x = h_a2
    if l >= 3:
        h_conv3 = conv2_2d(x, W_en_conv3) + b_en_conv3
        h_conv_bn_3 = input_data.batch_norm(h_conv3, f_num[3], phase_train)
        h_a3 = tf.nn.relu(h_conv_bn_3)
        h_pool3 = tf.nn.max_pool(h_a3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 3 in max_pool:
            x = h_pool3
        else:
            x = h_a3
    if l >= 4:
        h_conv4 = conv2d(x, W_en_conv4) + b_en_conv4
        h_conv_bn_4 = input_data.batch_norm(h_conv4, f_num[4], phase_train)
        h_a4 = tf.nn.relu(h_conv_bn_4)
        h_pool4 = tf.nn.max_pool(h_a4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 4 in max_pool:
            x = h_pool4
        else:
            x = h_a4
    if l >= 5:
        h_conv5 = conv2d(x, W_en_conv5) + b_en_conv5
        h_conv_bn_5 = input_data.batch_norm(h_conv5, f_num[5], phase_train)
        h_a5 = tf.nn.relu(h_conv_bn_5)
        h_pool5 = tf.nn.max_pool(h_a5, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 5 in max_pool:
            x = h_pool5
        else:
            x = h_a5
    if l >= 6:
        h_conv6 = conv2_2d(x, W_en_conv6) + b_en_conv6
        h_conv_bn_6 = input_data.batch_norm(h_conv6, f_num[6], phase_train)
        h_a6 = tf.nn.relu(h_conv_bn_6)
        h_pool6 = tf.nn.max_pool(h_a6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 6 in max_pool:
            x = h_pool6
        else:
            x = h_a6
    if l >= 7:
        h_conv7 = conv2d(x, W_en_conv7) + b_en_conv7
        h_conv_bn_7 = input_data.batch_norm(h_conv7, f_num[7], phase_train)
        h_a7 = tf.nn.relu(h_conv_bn_7)
        h_pool7 = tf.nn.max_pool(h_a7, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        if 7 in max_pool:
            x = h_pool7
        else:
            x = h_a7
    return x

def decoder(l, x):
    if l >= 7: 
        h_conv7 = conv2d(x, W_de_conv7) + b_de_conv7
        h_conv_bn_7 = input_data.batch_norm(h_conv7, f_num[6], phase_train)
        h_a7 = tf.nn.relu(h_conv_bn_7)
        b, h, w, c = h_a7.get_shape().as_list()
        h_upsamp7 = tf.image.resize_images(h_a7, (h * 2, w * 2))
        if 7 in max_pool:
            x = h_upsamp7
        else:
            x = h_a7
    if l >= 6: 
        h_conv6 = conv2_2d(x, W_de_conv6) + b_de_conv6
        h_conv_bn_6 = input_data.batch_norm(h_conv6, f_num[5], phase_train)
        h_a6 = tf.nn.relu(h_conv_bn_6)
        b, h, w, c = h_a6.get_shape().as_list()
        h_upsamp6 = tf.image.resize_images(h_a6, (h * 2, w * 2))
        if 6 in max_pool:
            x = h_upsamp6
        else:
            x = h_a6
    if l >= 5: 
        h_conv5 = conv2d(x, W_de_conv5) + b_de_conv5
        h_conv_bn_5 = input_data.batch_norm(h_conv5, f_num[4], phase_train)
        h_a5 = tf.nn.relu(h_conv_bn_5)
        b, h, w, c = h_a5.get_shape().as_list()
        h_upsamp5 = tf.image.resize_images(h_a5, (h * 2, w * 2))
        if 5 in max_pool:
            x = h_upsamp5
        else:
            x = h_a5
    if l >= 4: 
        h_conv4 = conv2d(x, W_de_conv4) + b_de_conv4
        h_conv_bn_4 = input_data.batch_norm(h_conv4, f_num[3], phase_train)
        h_a4 = tf.nn.relu(h_conv_bn_4)
        b, h, w, c = h_a4.get_shape().as_list()
        h_upsamp4 = tf.image.resize_images(h_a4, (h * 2, w * 2))
        if 4 in max_pool:
            x = h_upsamp4
        else:
            x = h_a4
    if l >= 3:    
        h_conv3 = conv2_2d(x, W_de_conv3) + b_de_conv3
        h_conv_bn_3 = input_data.batch_norm(h_conv3, f_num[2], phase_train)
        h_a3 = tf.nn.relu(h_conv_bn_3)
        b, h, w, c = h_a3.get_shape().as_list()
        h_upsamp3 = tf.image.resize_images(h_a3, (h * 2, w * 2))
        if 3 in max_pool:
            x = h_upsamp3
        else:
            x = h_a3
    if l >= 2:
        h_conv2 = conv2d(x, W_de_conv2) + b_de_conv2
        h_conv_bn_2 = input_data.batch_norm(h_conv2, f_num[1], phase_train)
        h_a2 = tf.nn.relu(h_conv_bn_2)
        b, h, w, c = h_a2.get_shape().as_list()
        h_upsamp2 = tf.image.resize_images(h_a2, (h * 2, w * 2))
        if 2 in max_pool:
            x = h_upsamp2
        else:
            x = h_a2
    if l >= 1:
        h_conv1 = conv2d(x, W_de_conv1) + b_de_conv1
        h_conv_bn_1 = input_data.batch_norm(h_conv1, f_num[0], phase_train)
        h_a1 = tf.nn.relu(h_conv_bn_1)
        b, h, w, c = h_a1.get_shape().as_list()
        h_upsamp1 = tf.image.resize_images(h_a1, (h * 2, w * 2))
        if 1 in max_pool:
            x = h_upsamp1
        else:
            x = h_a1
    return x

#---network parameter---#
X = tf.placeholder(tf.float32, shape = (None, n_input))
y_ = tf.placeholder(tf.float32, shape = (None, 10))
sess.run(tf.initialize_all_variables())
#  preprocessing data
dataset = input_data.CIFAR10(sys.argv[1])
labeled_image, label, train_image, train_label, validate_image, validate_label = dataset.labeled_image()
train_image /= 255.
validate_image /= 255.
unlabeled_image = dataset.unlabeled_image()
all_image = np.concatenate((labeled_image, unlabeled_image), axis = 0) / 255.
batch_size = 50
batch = input_data.minibatch(all_image, batch_size = batch_size)
labeled_image /= 255.
unlabeled_image /= 255.
for l in range(1, L + 1, 1):
#---construct model---#
    encoder_op = encoder(l, X)
    print encoder_op.get_shape()
    #  autoencoder model
    decoder_op = tf.reshape(decoder(l, encoder_op), [-1, n_input])
    y_pred = decoder_op
    y_true = X
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
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
    e = 5 * l
    for epoch in range(e):
        for i in range(batch.shape[0]):
            _, c, y_p, y_t = sess.run([optimizer, cost, y_pred, y_true], feed_dict = {X: all_image[batch[i]], phase_train: True})
            print 'fine tune, layer ' + str(l) + '/' + str(L)
            print 'epoch ' + str(epoch + 1) + '/'+ str(e) + ' batch '  + str(i + 1) + '/' + str(batch.shape[0]) +' cost:' + str(c) 

saver = tf.train.Saver({
    'W1': W_en_conv1, 'b1': b_en_conv1,
    'W2': W_en_conv2, 'b2': b_en_conv2, 
    #  'W3': W_en_conv3, 'b3': b_en_conv3, 
    #  'W4': W_en_conv4, 'b4': b_en_conv4, 
    #  'W5': W_en_conv5, 'b5': b_en_conv5,
    #  'W6': W_en_conv6, 'b6': b_en_conv6,
    #  'W7': W_en_conv7, 'b7': b_en_conv7,
    })
saver.save(sess, "./pretrain.ckpt")
