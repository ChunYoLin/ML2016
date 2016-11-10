import pickle as pk
import numpy as np
import tensorflow as tf
import input_data
import time

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

#---partition dataset into train and validation set---#
with open('./image.pk', 'rb') as im, open('./label.pk') as l:
    labeled_image = pk.load(im)
    label = pk.load(l)
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
train_image = np.concatenate((train_image, labeled_image[5000:]), axis = 0) 
train_label = np.concatenate((train_label, label[5000:]), axis = 0)
validate_image = np.asarray(validate_image)
validate_label = np.asarray(validate_label)
#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#---model definition---#
sess = tf.InteractiveSession(config = config)
#  input layer and ouput layer
x = tf.placeholder(tf.float32, shape = (None, 3072))
y_ = tf.placeholder(tf.float32, shape = (None, 10))
keep_prob = tf.placeholder(tf.float32)
#  conv layer 1
W_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 3, 96], stddev = 5e-2))
b_conv1 = tf.Variable(tf.constant(value = 0., shape = [96]))
x_image = tf.reshape(x, [-1, 32, 32, 3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
h_drop1 = tf.nn.dropout(h_pool1, keep_prob)
#  conv layer2
W_conv2 = tf.Variable(tf.truncated_normal(shape = [3, 3, 96, 96], stddev = 5e-2))
b_conv2 = tf.Variable(tf.constant(value = 0., shape = [96]))
h_conv2 = tf.nn.relu(conv2d(h_drop1, W_conv2) + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
h_drop2 = tf.nn.dropout(h_pool2, keep_prob)
#  conv layer3
W_conv3 = tf.Variable(tf.truncated_normal(shape = [3, 3, 96, 192], stddev = 5e-2))
b_conv3 = tf.Variable(tf.constant(value = 0., shape = [192]))
h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3) + b_conv3)
h_drop3 = tf.nn.dropout(h_conv3, keep_prob)
#  conv layer4
W_conv4 = tf.Variable(tf.truncated_normal(shape = [1, 1, 192, 192], stddev = 5e-2))
b_conv4 = tf.Variable(tf.constant(value = 0., shape = [192]))
h_conv4 = tf.nn.relu(conv2d(h_drop3, W_conv4) + b_conv4)
h_drop4 = tf.nn.dropout(h_conv4, keep_prob)
#  conv layer5
W_conv5 = tf.Variable(tf.truncated_normal(shape = [1, 1, 192, 10], stddev = 5e-2))
b_conv5 = tf.Variable(tf.constant(value = 0., shape = [10]))
h_conv5 = tf.nn.relu(conv2d(h_drop4, W_conv5) + b_conv5)
h_drop5 = tf.nn.dropout(h_conv5, keep_prob)
h_pool5 = tf.nn.avg_pool(h_drop5, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
y_conv = tf.reshape(h_pool5, [-1, 10])

#  #  fully connected layer 1
#  W_fc1 = tf.Variable(tf.truncated_normal(shape = [8 * 8 * 10, 1024], stddev = 1 / 1024.))
#  b_fc1 = tf.Variable(tf.constant(value = 0.1, shape = [1024]))
#  h_drop5_flat = tf.reshape(h_drop5, [-1, 8 * 8 * 10])
#  h_fc1 = tf.nn.relu(tf.matmul(h_drop5_flat, W_fc1) + b_fc1)
#  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#  #  output layer
#  W_fc2 = tf.Variable(tf.random_normal(shape = [1024, 10], stddev = 0.01))
#  b_fc2 = tf.Variable(tf.constant(value = 0., shape = [10]))
#  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#---training initial---#
#  define loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#  adam optimizer
train_step = tf.train.AdamOptimizer(3e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
#  minibatch  
batch_size = 100
batch = input_data.minibatch(train_image, batch_size = batch_size)
#  training
for epoch in range(2000):
    loss = 0.
    acc = 0.
    for j in range(batch.shape[0]):
        loss_val, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict = {x: train_image[batch[j]], y_: train_label[batch[j]], keep_prob: 1.0})
        loss += loss_val / batch.shape[0]
        acc += train_accuracy / batch.shape[0]
        sess.run(train_step, feed_dict = {x: train_image[batch[j]], y_: train_label[batch[j]], keep_prob: 0.6})
    print "epoch %d, loss %g, training accuracy %g"%(epoch, loss, acc)
    #  validation
    print "validation set accuracy", sess.run(accuracy, feed_dict = {x: validate_image, y_: validate_label, keep_prob: 1.0})


#---testing initial---#
y_conv_softmax = tf.nn.softmax(y_conv)
test_batch_result = tf.argmax(y_conv_softmax, 1)
test_all_result = []
#  testing
dataset = input_data.CIFAR10()
test_image = dataset.test_image() / 255.
for i in range(100):
    batch_result = sess.run(test_batch_result, feed_dict = {x: test_image[i * 100 : (i + 1) * 100], keep_prob: 1.0})
    test_all_result.append(batch_result)
test_all_result = np.asarray(test_all_result).reshape(-1)
#  output the testing result
with open('pred.csv', 'w') as result_file:
    result_file.write('ID,class\n')
    for i in range(test_all_result.shape[0]):
        result_file.write(str(i) + ',' + str(test_all_result[i]) + '\n')
