import cPickle as pk
import numpy as np
import tensorflow as tf
import input_data
import time
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

with tf.device('/cpu'):
#---partition dataset into train and validation set---#
    with open('./train_code', 'r') as f1, open('./validate_code') as f2:
        train_code = pk.load(f1)
        validate_code = pk.load(f2)
    dataset = input_data.CIFAR10()
    labeled_image, label, train_image, train_label, validate_image, validate_label = dataset.labeled_image()
    labeled_image_flip = dataset.labeled_image_flip() / 255.
    labeled_image /= 255.
    train_image /= 255.
    validate_image /= 255.
    train_image = np.concatenate((train_image, labeled_image_flip.reshape(-1, 3072)), axis = 0)
    train_label = np.concatenate((train_label, label), axis = 0)
    #  allow gpu memory growth        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #---model definition---#
    sess = tf.InteractiveSession(config = config)
    #  input layer and ouput layer
    keep_prob_in = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, shape = (None, 2048))
    x = tf.nn.dropout(x, keep_prob_in)
    y_ = tf.placeholder(tf.float32, shape = (None, 10))
    #  dropout rate
    keep_prob = tf.placeholder(tf.float32)
    #  conv layer 1
    W_conv1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 8, 192], stddev = 5e-2))
    b_conv1 = tf.Variable(tf.constant(value = 0.1, shape = [192]))
    x_image = tf.reshape(x, [-1, 16, 16, 8])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)
    #  fully connected layer 1
    W_fc1 = tf.Variable(tf.truncated_normal(shape = [8 * 8 * 192, 1024], stddev = 1 / 1024.))
    b_fc1 = tf.Variable(tf.constant(value = 0.1, shape = [1024]))
    h_drop1_flat = tf.reshape(h_drop1, [-1, 8 * 8 * 192])
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
    sess.run(tf.initialize_all_variables())
    for k in range(3):
        #  minibatch  
        batch_size = 100
        if k == 0:
            batch = input_data.minibatch(train_code, batch_size = batch_size)
        else:
            batch = input_data.minibatch(self_labeled_image, batch_size = batch_size)
        #  training
        if k == 0:
            for i in range(200):
                loss = 0.
                acc = 0.
                for j in range(batch.shape[0]):
                    loss_val, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict = {x: train_code[batch[j]], y_: train_label[batch[j]], keep_prob_in: 1.0, keep_prob: 1.0})
                    loss += loss_val / batch.shape[0]
                    acc += train_accuracy / batch.shape[0]
                    sess.run(train_step, feed_dict = {x: train_code[batch[j]], y_: train_label[batch[j]], keep_prob_in: 0.8, keep_prob: 0.5})
                print "self_training:", k
                print "stage 1: labeled training"
                print "epoch %d, loss %g, training accuracy %g"%(i, loss, acc)
                #  validation
                print "self_training:", k
                print "validation set accuracy", sess.run(accuracy, feed_dict = {x: validate_code, y_: validate_label, keep_prob_in: 1.0, keep_prob: 1.0})
        else:
            for i in range(30):
                loss = 0.
                acc = 0.
                for j in range(batch.shape[0]):
                    loss_val, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict = {x: self_labeled_image[batch[j]], y_: self_label[batch[j]], keep_prob_in: 1.0, keep_prob: 1.0})
                    loss += loss_val / batch.shape[0]
                    acc += train_accuracy / batch.shape[0]
                    sess.run(train_step, feed_dict = {x: self_labeled_image[batch[j]], y_: self_label[batch[j]], keep_prob_in: 0.8, keep_prob: 0.5})
                print "self_training:", k
                print "stage 2: add self labeled training"
                print "epoch %d, loss %g, training accuracy %g"%(i, loss, acc)
                #  validation
                print "self_training:", k
                print "validation set accuracy", sess.run(accuracy, feed_dict = {x: validate_image, y_: validate_label, keep_prob_in: 1.0, keep_prob: 1.0})

        #---unlabeled testing initial---#
        unlabeled_image = dataset.unlabeled_image() / 255.
        y_conv_softmax = tf.nn.softmax(y_conv)
        unlabeled_batch_result = tf.argmax(y_conv_softmax, 1)
        unlabeled_all_softmax_result = []
        unlabeled_all_argmax_result = []
        #  unlabeled testing
        print "self_training:", k
        print "unlabeled image self labeling...."
        for i in range(100):
            batch_softmax_result, batch_argmax_result = sess.run([y_conv_softmax, unlabeled_batch_result], feed_dict = {x: unlabeled_image[i * 450 : (i + 1) * 450], keep_prob: 1.0})
            unlabeled_all_softmax_result.append(batch_softmax_result)
            unlabeled_all_argmax_result.append(batch_argmax_result)
        unlabeled_all_softmax_result = np.asarray(unlabeled_all_softmax_result).reshape(-1, 10)
        index = np.amax(unlabeled_all_softmax_result, axis = 1) > 0.9
        unlabeled_all_argmax_result = np.asarray(unlabeled_all_argmax_result).reshape(-1)
        hard_label = np.zeros(shape = unlabeled_all_softmax_result.shape)
        for i in range(hard_label.shape[0]):
            hard_label[i, unlabeled_all_argmax_result[i]] = 1.
        self_labeled_image = np.concatenate((train_image, unlabeled_image[index]), axis = 0)
        self_label = np.concatenate((train_label, hard_label[index]), axis = 0)

    #---testing initial---#
    y_conv_softmax = tf.nn.softmax(y_conv)
    test_batch_result = tf.argmax(y_conv_softmax, 1)
    test_all_result = []
    #  testing
    test_image = dataset.test_image() / 255.
    print 'testing...'
    for i in range(100):
        batch_result = sess.run(test_batch_result, feed_dict = {x: test_image[i * 100 : (i + 1) * 100], keep_prob: 1.0})
        test_all_result.append(batch_result)
    test_all_result = np.asarray(test_all_result).reshape(-1)
    #  output the testing result
    with open('pred.csv', 'w') as result_file:
        result_file.write('ID,class\n')
        for i in range(test_all_result.shape[0]):
            result_file.write(str(i) + ',' + str(test_all_result[i]) + '\n')
