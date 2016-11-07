import numpy as np
import tensorflow as tf
import input_data
import scipy.spatial.distance
import cPickle as pk

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
#---network parameter---#
n_input = 3072

X = tf.placeholder(tf.float32, shape = (None, n_input))
#  encode_layer
W_en_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 3, 64], stddev = 5e-2))
b_en_conv1 = tf.Variable(tf.constant(value = 0.1, shape = [64]))
W_en_conv2 = tf.Variable(tf.truncated_normal(shape = [3, 3, 64, 16], stddev = 5e-2))
b_en_conv2 = tf.Variable(tf.constant(value = 0.1, shape = [16]))
W_en_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 16, 16], stddev = 5e-2))
b_en_conv3 = tf.Variable(tf.constant(value = 0.1, shape = [16]))
#  decode layer
W_de_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 16, 16], stddev = 5e-2))
b_de_conv1 = tf.Variable(tf.constant(value = 0.1, shape = [16]))
W_de_conv2 = tf.Variable(tf.truncated_normal(shape = [3, 3, 16, 64], stddev = 5e-2))
b_de_conv2 = tf.Variable(tf.constant(value = 0.1, shape = [64]))
W_de_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 64, 3], stddev = 5e-2))
b_de_conv3 = tf.Variable(tf.constant(value = 0.1, shape = [3]))
#---building encoder and decoder---#
def encoder(x):
    x_image = tf.reshape(x, [-1, 32, 32, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_en_conv1) + b_en_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_en_conv2) + b_en_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_en_conv3) + b_en_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return h_pool3

def decoder(x):
    h_conv1 = tf.nn.relu(conv2d(x, W_de_conv1) + b_de_conv1)
    b, h, w, c = h_conv1.get_shape().as_list()
    h_upsamp1 = tf.image.resize_images(h_conv1, (h * 2, w * 2))

    h_conv2 = tf.nn.relu(conv2d(h_upsamp1, W_de_conv2) + b_de_conv2)
    b, h, w, c = h_conv2.get_shape().as_list()
    h_upsamp2 = tf.image.resize_images(h_conv2, (h * 2, w * 2))
    
    h_conv3 = tf.nn.sigmoid(conv2d(h_upsamp2, W_de_conv3) + b_de_conv3)
    h_upsamp3 = tf.image.resize_images(h_conv3, (32, 32))
    h_upsamp3 = tf.reshape(h_upsamp3, [-1, 3072])
    return h_upsamp3

#---construct model---#
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
#  prediction
y_pred = decoder_op
y_true = X
#  define loss
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
#  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))
optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)
init = tf.initialize_all_variables()
#  preprocessing data
dataset = input_data.CIFAR10()
labeled_image, label = dataset.labeled_image()
unlabeled_image = dataset.unlabeled_image()
all_image = np.concatenate((labeled_image, unlabeled_image), axis = 0) / 255.
batch_size = 200
batch = input_data.minibatch(all_image, batch_size = batch_size)

#---run the graph---#
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for i in range(batch.shape[0]):
            _, c, y_p, y_t = sess.run([optimizer, cost, y_pred, y_true], feed_dict = {X: all_image[batch[i]]})
        print 'epoch:' + str(epoch) + ' cost:' + str(c) 
        print 'y_pred:' + str(y_p) 
        print 'y_true:' + str(y_t)

#---encode label and unlabel image---#
    label_code = sess.run(encoder_op, feed_dict = {X: labeled_image})
    unlabel_code = []
    for i in range(100):
        code = sess.run(encoder_op, feed_dict = {X: unlabeled_image[i * 450 : (i + 1) * 450]})
        unlabel_code.append(code)

    unlabel_code = np.asarray(unlabel_code).reshape(45000, -1)
    label_code = label_code.reshape(5000, -1)
    print label_code.shape, unlabel_code.shape

#---label unlabel_image by encode_code cosine simiularity---#
    self_label = np.ndarray(shape = (unlabel_code.shape[0], 10))
unlabel_simularity = np.argmin(scipy.spatial.distance.cdist(unlabel_code, label_code, 'cosine'), axis = 1)
for i in range(self_label.shape[0]):
    self_label[i] = label[unlabel_simularity[i]]

labeled_image = np.concatenate((labeled_image, unlabeled_image), axis = 0)
label = np.concatenate((label, self_label), axis = 0)
print 'dump the code'

with open('image.pk', 'w') as f:
    pk.dump(labeled_image, f)
with open('label.pk', 'w') as f:
    pk.dump(label, f)


