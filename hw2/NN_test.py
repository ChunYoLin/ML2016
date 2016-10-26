import numpy as np
import re
import pickle
import sys

def sigmoid(x):
    try:
        return 1. / (1 + np.exp(-x))
    except OverflowError:
        return 0.
test_data = open(sys.argv[2], 'r')
w_file = open(sys.argv[1], 'r')
x_mean = pickle.load(w_file)
x_std = pickle.load(w_file)
w = pickle.load(w_file)
w_file.close()

x = []
for row in test_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[6 : 1 + 55])
x = np.asarray(x, dtype = np.float32)
x = (x - x_mean) / x_std
L = 5
s = [x.shape[1], 38, 26, 18, 1]
a = [[] for i in range(L)]
a[0] = x
biasa = np.ones(shape = (a[0].shape[0], 1))
a[0] = np.concatenate((biasa, x), axis = 1)
for l in range(1, L):
    a[l] = sigmoid(np.dot(a[l - 1], w[l - 1]))
    if l != L - 1:
        biasa = np.ones(shape = (a[l].shape[0], 1))
        a[l] = np.concatenate((biasa, a[l]), axis = 1)
y_ = a[L - 1]
pred = open(sys.argv[3], 'w')
pred.write('id,label\n')
for i in range(y_.shape[0]):
    if y_[i] > 0.5:
        pred.write(str(i + 1) + ',' + str(1) + '\n')
        y_[i] = 1
    else:
        pred.write(str(i + 1) + ',' + str(0) + '\n')
        y_[i] = 0
pred.close()
