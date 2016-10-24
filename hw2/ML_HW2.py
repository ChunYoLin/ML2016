import numpy as np
import re

train_data = open('./spam_data/spam_train.csv', 'r')
x = []
y = []
for row in train_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[1 : 1 + 56])
    y.append(row_l[len(row_l) - 1])
x = np.asarray(x, dtype = np.float32)
bias = np.ones(shape = (x.shape[0], 1))
x = np.concatenate((bias, x), axis = 1)
w = np.zeros(shape = x.shape[1])
y = np.asarray(y, dtype = np.float32)
m = x.shape[0]
Learning_rate = 0.5 
gw_his = 0.
k = 0
while True:
    y_ = 1. / (1. + np.exp(((-1.) * np.dot(x, w))))
    for i in range(y_.shape[0]):
        if y_[i] == 1:
            y_[i] -= 0.000000000001
    L = (1. / m) * np.sum((-y) * np.log(y_) - (1. - y) * np.log(1. - y_))
    gw = -np.dot(x.transpose(), (y - y_))
    gw_his += gw**2
    gw_mean = np.mean(np.abs(gw))
    print 'iter', k
    print 'Learning_rate', Learning_rate
    print 'cross entrophy', L
    print 'mean of gradient', gw_mean
    w -= (Learning_rate / np.sum(gw_his, axis = 0)**0.5) * gw 
    if gw_mean < 0.04:
        w_file_name = './spam.weights'
        w_file = open(w_file_name, 'w')
        for i in range(len(w)):
            w_file.write(str(w[i]))
            w_file.write(',')
        w_file.close()
        break
    k += 1
