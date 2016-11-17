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
def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name = 'beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name = 'gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name = 'moments')
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def label_classifier(X):
    print "classify data shape", X.shape
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
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    return label_out

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
        return labeled_image, label, train_image, train_label, validate_image, validate_label
    def image_flip(self, image):
        #  all_label = pk.load(open('./data/all_label.p', 'rb'))
        #  all_label = np.asarray(all_label, dtype = np.float32)
        image = image.reshape(-1, 3, 32, 32)
        flip_image = np.zeros_like(image)
        for i in range(flip_image.shape[0]):
            for j in range(3):
                flip_image[i, j] = np.fliplr(image[i, j])
        return flip_image
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
        
