import sys
import re
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

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

with open('./reduce_titleV2.txt', 'r') as f:
    word_set = {}
    w_id = 0
    title = []
    for TITLE in f:
        title.append(TITLE)
        for w in re.split('\W+', TITLE):
            if w.lower() not in word_set and w != '':
                word_set[w.lower()] = w_id
                w_id += 1
title_wd = np.zeros(shape = (len(title), len(word_set)))
print len(word_set)
print len(title)
for idx, t in enumerate(title):
    for w in re.split('\W+', t):
        if w != '':
            title_wd[idx, word_set[w.lower()]] += 1

idf = np.zeros(shape = len(word_set))
for i in range(len(word_set)):
    for j in range(len(title)):
        if title_wd[j, i] > 0:
            idf[i] += 1.
idf = len(title) / idf
title_wd *= np.log(idf)
#  input
X = tf.placeholder(tf.float32, shape = [None, len(word_set)])
l_num = 5
f_num = [len(word_set), 2000, 500, 250, 125, 10]

W_en = [0]
b_en = [0]
h_en = [X]
a_en = [X]
for i in range(1, l_num + 1, 1):
    W_en.append(tf.Variable(tf.truncated_normal(shape = [f_num[i - 1], f_num[i]], stddev = 0.01)))
    b_en.append(tf.Variable(tf.constant(value = 0.1, shape = [f_num[i]])))
    h_en.append(tf.matmul(a_en[i - 1], W_en[i]) + b_en[i])
    a_en.append(tf.nn.relu(h_en[i]))
    print a_en[i - 1].get_shape(), a_en[i].get_shape()

W_de = [0]
b_de = [0]
h_de = [a_en[l_num]]
a_de = [a_en[l_num]]
for i in range(1, l_num + 1, 1):
    W_de.append(tf.Variable(tf.truncated_normal(shape = [f_num[l_num - (i - 1)], f_num[l_num - i]], stddev = 0.01)))
    b_de.append(tf.Variable(tf.constant(value = 0.1, shape = [f_num[l_num - i]])))
    h_de.append(tf.matmul(a_de[i - 1], W_de[i]) + b_de[i])
    a_de.append(tf.nn.relu(h_de[i]))
    print a_de[i - 1].get_shape(), a_de[i].get_shape()

y_pred = a_de[l_num]
y_true = X
encoder = h_en[l_num]
cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

batch = minibatch(title_wd)
e = 50
for epoch in range(e):
    for b_id in range(batch.shape[0]):
        _, loss = sess.run([train_step, cost], feed_dict = {X:title_wd[batch[b_id]]})    
        print sess.run(a_de[l_num], feed_dict = {X:title_wd[0:1]})
        print title_wd[0:1]
        print 'epoch %d / %d, batch %d / %d, loss %g'%(epoch + 1, e, b_id + 1, batch.shape[0], loss)
code = []
for i in range(200):
    code.append(sess.run(encoder, feed_dict = {X:title_wd[i * 100 : (i+1) * 100]}))
code = np.asarray(code).reshape(len(title), -1)
#  print code[0:50]
Group = KMeans(n_clusters = 20, random_state = 0).fit_predict(code)

with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')

