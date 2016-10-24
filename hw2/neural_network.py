import numpy as np
import re


train_data = open('./spam_data/spam_train.csv', 'r')
x_ = []
y = []
for row in train_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x_.append(row_l[1 : 1 + 56])
    y.append(row_l[len(row_l) - 1])

x_ = np.asarray(x_, dtype = np.float32)
x_mean = np.mean(x_, axis = 0)
x_std = np.std(x_, axis = 0)
x_ = (x_ - np.mean(x_, axis = 0)) / np.std(x_, axis = 0)

y = np.asarray(y, dtype = np.float32)
y = y.reshape(y.shape[0], 1)
L = 4
s = [x_.shape[1], 4, 2, 1]

w = [[] for i in range(L - 1)]
DELTA = [[] for i in range(L - 1)]
DELTA_his = [[] for i in range(L - 1)]
m = [[] for i in range(L - 1)]
v = [[] for i in range(L - 1)]
for l in range(len(w)):
    w[l] = np.full((s[l] + 1, s[l + 1]), 0.0001)
for l in range(len(DELTA)):
    DELTA[l] = np.zeros_like(w[l])
    DELTA_his[l] = np.zeros_like(w[l])
    m[l] = np.zeros_like(w[l])
    v[l] = np.zeros_like(w[l])
a = [[] for i in range(L)]
delta = [[] for i in range(L)]
k = x_.shape[0]
Learning_rate = 0.008
ITER = 0
BETA1 = 0.9
BETA2 = 0.999
E = 10**(-8)
t = 0
while(True):
    t += 1
    a[0] = x_
    biasa = np.ones(shape = (a[0].shape[0], 1))
    a[0] = np.concatenate((biasa, x_), axis = 1)
    for l in range(1, L):
        a[l] = 1. / (1. + np.exp(((-1.) * np.dot(a[l - 1], w[l - 1]))))
        if l != L - 1:
            biasa = np.ones(shape = (a[l].shape[0], 1))
            a[l] = np.concatenate((biasa, a[l]), axis = 1)
    delta[L - 1] = a[L - 1] - y
    for l in range(L - 2, 0, -1):
        if l != L - 2:
            delta[l] = np.dot(delta[l + 1][:, 1:], w[l].transpose()) * a[l] * (1 - a[l])
        else:
            delta[l] = np.dot(delta[l + 1], w[l].transpose()) * a[l] * (1 - a[l])
    for l in range(L - 1):
        if l != L - 2:
            DELTA[l] = np.dot(a[l].transpose(), delta[l + 1][:, 1:])
        else:
            DELTA[l] = np.dot(a[l].transpose(), delta[l + 1])
        m[l] = BETA1 * m[l] + (1 - BETA1) * DELTA[l]
        v[l] = BETA2 * v[l] + (1 - BETA2) * DELTA[l]**2
        m_hat = m[l] / (1 - BETA1**t)
        v_hat = v[l] / (1 - BETA2**t)
        w[l] -= Learning_rate * m_hat / (v_hat**0.5 + E)
        #  DELTA_his[l] += DELTA[l]**2
        #  w[l] -= (Learning_rate  / np.sum(DELTA_his[l], axis = 0)**0.5) * DELTA[l]
    acc = 0.
    for i in range(k):
        if (y[i] == 1 and a[L - 1][i] > 0.5) or (y[i] == 0 and a[L - 1][i] <= 0.5):
            acc += 1
    print "iter" + str(ITER) + " loss " + str(np.mean(np.abs(delta[L - 1]))) + " acc " + str(acc / k)
    ITER += 1
    if ITER == 5000:
        break
 
test_data = open('./spam_data/spam_test.csv', 'r')
x = []
for row in test_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[1 : 1 + 56])
x = np.asarray(x, dtype = np.float32)
x = (x - x_mean) / x_std
a[0] = x
biasa = np.ones(shape = (a[0].shape[0], 1))
a[0] = np.concatenate((biasa, x), axis = 1)
for l in range(1, L):
    a[l] = 1. / (1. + np.exp(((-1.) * np.dot(a[l - 1], w[l - 1]))))
    if l != L - 1:
        biasa = np.ones(shape = (a[l].shape[0], 1))
        a[l] = np.concatenate((biasa, a[l]), axis = 1)
y_ = a[L - 1]
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
