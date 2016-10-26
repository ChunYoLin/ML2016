import numpy as np
import re
import sys
train_data = open(sys.argv[1], 'r')
feature = []
x = []
y = []
for row in train_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[1 : 1 + 56])
    y.append(row_l[len(row_l) - 1])
x = np.asarray(x, dtype = np.float32)
bias = np.ones(shape = (x.shape[0], 1))
x = np.concatenate((bias, x), axis = 1)
w = np.random.normal(0., 0.01, (x.shape[1]))
y = np.asarray(y, dtype = np.float32)
k = x.shape[0]
Learning_rate = 0.005
gw_his = 0.
t = 0
BETA1 = 0.9
BETA2 = 0.999
E = 10**(-8)
m = np.zeros_like(w)
v = np.zeros_like(w)

while True:
    t += 1
    y_ = 1. / (1. + np.exp(((-1.) * np.dot(x, w))))
    acc = 0.
    for i in range(y_.shape[0]):
        if (y_[i] > 0.5 and y[i] == 1) or (y_[i] <= 0.5 and y[i] == 0):
            acc += 1
        if y_[i] == 1:
            y_[i] -= 0.000000000001
    L = (1. / k) * np.sum((-y) * np.log(y_) - (1. - y) * np.log(1. - y_))
    gw = -np.dot(x.transpose(), (y - y_))
    gw_mean = np.mean(np.abs(gw))
    print 'iter', t
    print 'Learning_rate', Learning_rate
    print 'cross entrophy', L
    print 'mean of gradient', gw_mean
    print 'acc', acc / y_.shape[0]
    m = BETA1 * m + (1 - BETA1) * gw
    v = BETA2 * v + (1 - BETA2) * gw**2
    m_hat = m / (1 - BETA1**t)
    v_hat = v / (1 - BETA2**t)
    w -= Learning_rate * m_hat / (v_hat**0.5 + E) 
    #  gw_mean = np.mean(np.abs(gw))
    #  w -= (Learning_rate / np.sum(gw_his, axis = 0)**0.5) * gw 
    if t == 4000:
        w_file_name = sys.argv[2]
        w_file = open(w_file_name, 'w')
        for i in range(len(w)):
            w_file.write(str(w[i]))
            w_file.write(',')
        w_file.close()
        break
    k += 1
