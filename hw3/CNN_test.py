import pickle as pk
import numpy as np
import tensorflow as tf
import input_data
import time
import sys

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def conv2_2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 2, 2, 1], padding = 'SAME')

dataset = input_data.CIFAR10(sys.argv[1])
labeled_image, label, train_image, train_label, validate_image, validate_label = dataset.labeled_image()
train_image /= 255.
validate_image /= 255.
labeled_image /= 255.
labeled_image_flip = dataset.image_flip(labeled_image).reshape(-1, 3072)
train_image = np.concatenate((labeled_image, labeled_image_flip), axis = 0)
train_label = np.concatenate((label, label), axis = 0)

#---partition dataset into train and validation set---#
dataset = input_data.CIFAR10(sys.argv[1])
#  allow gpu memory growth        
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#---model definition---#
sess = tf.InteractiveSession(config = config)
#  input layer and ouput layer
X = tf.placeholder(tf.float32, shape = (None, 3072))
y_ = tf.placeholder(tf.float32, shape = (None, 10))
phase_train = tf.placeholder(tf.bool, name = 'phase_train')
keep_prob = tf.placeholder(tf.float32)
keep_prob_in = tf.placeholder(tf.float32)
conv_l = 2
fc_l = 1
f_num = [3, 128, 256, 96, 192, 192, 192, 10]
f_size = [0, 3, 3, 3, 3, 3, 1, 1]
max_pool = [1, 2]
glb_avg = False
fc_node = [1024]

#  input layer
x_image = tf.reshape(X, [-1, 32, 32, 3])
x_image_drop = tf.nn.dropout(x_image, keep_prob_in)
x = x_image_drop
W_conv = [0]
b_conv = [0]
h_conv = [x_image]
h_a = [x_image]
h_drop = [x_image_drop]
#  conv layer
for i in range(1, conv_l + 1, 1):
    W_conv.append(tf.Variable(tf.truncated_normal(shape = [f_size[i], f_size[i], f_num[i - 1], f_num[i]], stddev = 5e-2)))
    b_conv.append(tf.Variable(tf.constant(value = 0.1, shape = [f_num[i]])))
    if i == 3 or i == 6:
        h_conv.append(conv2_2d(h_drop[i - 1], W_conv[i]) + b_conv[i])
    else:
        h_conv.append(conv2d(h_drop[i - 1], W_conv[i]) + b_conv[i])
    h_conv_bn = input_data.batch_norm(h_conv[i], f_num[i], phase_train)
    h_a.append(tf.nn.relu(h_conv_bn))
    if i in max_pool:
        pool = tf.nn.max_pool(h_a[i], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        h_drop.append(tf.nn.dropout(pool, keep_prob))
    elif glb_avg == True and i == conv_l:
        s = 32 / 2**len(max_pool)
        glb_pool = tf.nn.avg_pool(h_conv_bn, ksize = [1, s, s, 1], strides = [1, s, s, 1], padding = 'SAME')
        h_drop.append(tf.nn.dropout(glb_pool, keep_prob))
    else:
        h_drop.append(tf.nn.dropout(h_a[i], keep_prob))
    print str(h_drop[i - 1].get_shape()) + ' --> ' + str(h_drop[i].get_shape())

#  y_conv = tf.reshape(h_drop[conv_l], [-1, f_num[conv_l]])
#  fully connected layer 
s = 32 / 2**len(max_pool)
W_fc1 = tf.Variable(tf.truncated_normal(shape = [s * s * f_num[conv_l], 1024], stddev = 1 / 1024.))
b_fc1 = tf.Variable(tf.constant(value = 0.1, shape = [1024]))
h_drop_flat = tf.reshape(h_drop[conv_l], [-1, s * s * f_num[conv_l]])
h_fc1 = tf.nn.relu(tf.matmul(h_drop_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#  output layer
W_fc2 = tf.Variable(tf.random_normal(shape = [1024, 10], stddev = 0.01))
b_fc2 = tf.Variable(tf.constant(value = 0., shape = [10]))
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv_softmax = tf.nn.softmax(y_conv)
test_batch_result = tf.argmax(y_conv_softmax, 1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#  adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
saver = tf.train.Saver({
    'W1': W_conv[1], 'b1': b_conv[1], 
    'W2': W_conv[2], 'b2': b_conv[2], 
    'W_fc1': W_fc1, 'b_fc1': b_fc1,
    'W_fc2': W_fc2, 'b_fc2': b_fc2,
    })
saver.restore(sess, sys.argv[2])
print 'loading the model.......'
e = 1
batch_size = 100
batch = input_data.minibatch(train_image, batch_size = batch_size)
for epoch in range(e):
    acc = 0.
    loss = 0.
    for j in range(batch.shape[0]):
        loss_val, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict = {X: train_image[batch[j]], y_: train_label[batch[j]], keep_prob_in: 1.0, keep_prob: 1.0, phase_train: False})
        acc += train_accuracy / batch.shape[0]
        loss += loss_val / batch.shape[0]
        sess.run(train_step, feed_dict = {X: train_image[batch[j]], y_: train_label[batch[j]], keep_prob_in: 1.0, keep_prob: 0.6, phase_train: True})
    #  validation
    #  vacc = sess.run(accuracy, feed_dict = {X: validate_image, y_: validate_label, keep_prob_in: 1.0, keep_prob: 1.0, phase_train: False})


#---testing initial---#
test_all_result = []
#  testing
test_image = dataset.test_image() / 255.
for i in range(100):
    batch_result = sess.run(test_batch_result, feed_dict = {X: test_image[i * 100 : (i + 1) * 100], keep_prob_in: 1.0, keep_prob: 1.0, phase_train: False})
    test_all_result.append(batch_result)
test_all_result = np.asarray(test_all_result).reshape(-1)
#  output the testing result
with open(sys.argv[3], 'w') as result_file:
    result_file.write('ID,class\n')
    for i in range(test_all_result.shape[0]):
        result_file.write(str(i) + ',' + str(test_all_result[i]) + '\n')
