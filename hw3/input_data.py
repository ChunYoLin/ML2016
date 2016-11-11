import cPickle as pk
import numpy as np
import tensorflow as tf

def minibatch(X, batch_size = 50, Shuffle = True):
    all_batch = np.arange(X.shape[0])
    all_size = X.shape[0]
    if Shuffle:
        np.random.shuffle(all_batch)
    batch = []
    for a in range(all_size / batch_size):
        single_batch = []
        for b in range(batch_size):
            single_batch.append(all_batch[a * batch_size + b])
        batch.append(single_batch)
    batch = np.asarray(batch)
    return batch

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')


def label_classifier(X):
    print X.shape
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
    label_out = tf.nn.softmax(y_conv)
    
    #  define loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    #  adam optimizer
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    saver.restore(sess, 'model')
    return sess.run(label_out, feed_dict = {x: X, keep_prob_in: 1., keep_prob: 1.})
class CIFAR10:
    def __init__(self):
        pass
    def labeled_image(self):
        #  loading data from pikcle file
        all_label = pk.load(open('./data/all_label.p', 'rb'))
        all_label = np.asarray(all_label, dtype = np.float32)
        labeled_image = all_label.reshape(all_label.shape[0] * all_label.shape[1], all_label.shape[2])
        label = np.zeros(shape = (labeled_image.shape[0], 10), dtype = np.float32)
        for i in range(10):
            label[i * 500 : i * 500 + 500, i] = 1
        return labeled_image, label
    def unlabeled_image(self):
        #  loading data from pikcle file
        all_unlabel = pk.load(open('./data/all_unlabel.p', 'rb'))
        unlabeled_image = np.asarray(all_unlabel, dtype = np.float32)
        return unlabeled_image
    def test_image(self):
        #  loading data from pikcle file
        test = pk.load(open('./data/test.p', 'rb'))
        test_image = np.asarray(test['data'], dtype = np.float32)
        return test_image
