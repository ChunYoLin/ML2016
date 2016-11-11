import tensorflow as tf
import numpy as np
import input_data
import time
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#---partition dataset into train and validation set---#
dataset = input_data.CIFAR10()
train_image = []
train_label = []
validate_image = []
validate_label = []
train_set_size = 500
labeled_image, label = dataset.labeled_image()
labeled_image /= 255.
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
#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#---model definition---#
sess = tf.InteractiveSession(config = config)
#  input layer and ouput layer
keep_prob_in = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape = (None, 3072))
x = tf.nn.dropout(x, keep_prob_in)
y_ = tf.placeholder(tf.float32, shape = (None, 10))
#  dropout rate
keep_prob = tf.placeholder(tf.float32)
#  conv layer 1
W_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 3, 192], stddev = 5e-2))
b_conv1 = tf.Variable(tf.constant(value = 0.1, shape = [192]))
x_image = tf.reshape(x, [-1, 32, 32, 3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
h_drop1 = tf.nn.dropout(h_pool1, keep_prob)
#  fully connected layer 1
W_fc1 = tf.Variable(tf.truncated_normal(shape = [16 * 16 * 192, 1024], stddev = 1 / 1024.))
b_fc1 = tf.Variable(tf.constant(value = 0.1, shape = [1024]))
h_drop1_flat = tf.reshape(h_drop1, [-1, 16 * 16 * 192])
h_fc1 = tf.nn.relu(tf.matmul(h_drop1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#  output layer
W_fc2 = tf.Variable(tf.random_normal(shape = [1024, 10], stddev = 0.01))
b_fc2 = tf.Variable(tf.constant(value = 0., shape = [10]))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#---training initial---#
#  define loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#  adam optimizer
train_step = tf.train.AdamOptimizer(2e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, 'model')

