import numpy as np
import re
test_data = open('./spam_data/spam_test.csv', 'r')
x = []
for row in test_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[1 : 1 + 56])
x = np.asarray(x, dtype = np.float32)
bias = np.ones(shape = (x.shape[0], 1))
x = np.concatenate((bias, x), axis = 1)
w = np.zeros(shape = x.shape[1])
weight_file = open('./spam.weights', 'r')
weight = weight_file.read().split(',')
weight.remove('')
for idx, value in enumerate(weight):
    w[idx] = float(value)
y_ = 1. / (1. + np.exp(((-1.) * np.dot(x, w))))
pred = open('./pred.csv', 'w')
pred.write('id,label\n')
for i in range(y_.shape[0]):
    if y_[i] > 0.5:
        pred.write(str(i + 1) + ',' + str(1) + '\n')
        y_[i] = 1
    else:
        pred.write(str(i + 1) + ',' + str(0) + '\n')
        y_[i] = 0
pred.close()
