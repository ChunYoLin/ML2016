import numpy as np
import re
import pickle
import sys

def sigmoid(x):
    try:
        return 1. / (1 + np.exp(-x))
    except OverflowError:
        return 0.

train_data = open(sys.argv[1], 'r')
x_ = []
y = []
for row in train_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x_.append(row_l[6 : 1 + 55])
    y.append(row_l[len(row_l) - 1])

x_ = np.asarray(x_, dtype = np.float32)
x_mean = np.mean(x_, axis = 0)
x_std = np.std(x_, axis = 0)
x_ = (x_ - np.mean(x_, axis = 0)) / np.std(x_, axis = 0)
y = np.asarray(y, dtype = np.float32)
y = y.reshape(y.shape[0], 1)
L = 5
s = [x_.shape[1], 38, 26, 18, 1]

w = [[] for i in range(L - 1)]
DELTA = [[] for i in range(L - 1)]
DELTA_his = [[] for i in range(L - 1)]
m = [[] for i in range(L - 1)]
v = [[] for i in range(L - 1)]
for l in range(len(w)):
    w[l] = np.full((s[l] + 1, s[l + 1]), -0.318)
for l in range(len(DELTA)):
    DELTA[l] = np.zeros_like(w[l])
    DELTA_his[l] = np.zeros_like(w[l])
    m[l] = np.zeros_like(w[l])
    v[l] = np.zeros_like(w[l])
a = [[] for i in range(L)]
a_ = [[] for i in range(L)]
delta = [[] for i in range(L)]
k = x_.shape[0]
#  k = 3500
Learning_rate = .0001
ITER = 0
BETA1 = 0.9
BETA2 = 0.999
E = 10**(-8)
t = 0
LAMBDA = 0.
THREASH = 0.5
while(True):
    t += 1
    a[0] = x_[: k]
    biasa = np.ones(shape = (a[0].shape[0], 1))
    a[0] = np.concatenate((biasa, a[0]), axis = 1)
    a_[0] = x_
    biasa = np.ones(shape = (a_[0].shape[0], 1))
    a_[0] = np.concatenate((biasa, a_[0]), axis = 1)
    loss = 0.
    gradient = 0.
    for l in range(1, L):
        a[l] = sigmoid(np.dot(a[l - 1], w[l - 1]))
        a_[l] = sigmoid(np.dot(a_[l - 1], w[l - 1]))
        if l != L - 1:
            biasa = np.ones(shape = (a[l].shape[0], 1))
            a[l] = np.concatenate((biasa, a[l]), axis = 1)
            biasa = np.ones(shape = (a_[l].shape[0], 1))
            a_[l] = np.concatenate((biasa, a_[l]), axis = 1)
    delta[L - 1] = a[L - 1] - y[: k]
    loss += np.mean(np.abs(delta[L - 1]))
    for l in range(L - 2, 0, -1):
        if l != L - 2:
            delta[l] = np.dot(delta[l + 1][:, 1:], w[l].transpose()) * a[l] * (1 - a[l])
        else:
            delta[l] = np.dot(delta[l + 1], w[l].transpose()) * a[l] * (1 - a[l])
        loss += np.mean(np.abs(delta[l]))
    for l in range(L - 1):
        if l != L - 2:
            DELTA[l] = (1. / k) * np.dot(a[l].transpose(), delta[l + 1][:, 1:])
            DELTA[l][1:] += LAMBDA * w[l][1:]
        else:
            DELTA[l] = (1. / k) * np.dot(a[l].transpose(), delta[l + 1]) + LAMBDA * w[l]
        gradient += np.mean(np.abs(DELTA[l]))
        m[l] = BETA1 * m[l] + (1 - BETA1) * DELTA[l]
        v[l] = BETA2 * v[l] + (1 - BETA2) * DELTA[l]**2
        m_hat = m[l] / (1 - BETA1**t)
        v_hat = v[l] / (1 - BETA2**t)
        w[l] -= Learning_rate * m_hat / (v_hat**0.5 + E)
        #  DELTA_his[l] += DELTA[l]**2
        #  w[l] -= (Learning_rate  / np.sum(DELTA_his[l], axis = 0)**0.5) * DELTA[l]
    acc_train = 0.
    acc_test = 0.
    for i in range(x_.shape[0]):
        if (y[i] == 1 and a_[L - 1][i] > THREASH) or (y[i] == 0 and a_[L - 1][i] <= THREASH):
            if i < k:
                acc_train += 1
            else:
                acc_test += 1

    print "iter" + str(ITER)
    print " loss " + str(loss)
    print " gradient " + str(gradient)
    print " acc_train " + str(acc_train / k)
    print " acc_test " + str(acc_test / (x_.shape[0] - k + 0.0000001))
    ITER += 1
    if ITER == 50000:
        break


w_file = open(sys.argv[2], 'w')
pickle.dump(x_mean, w_file)
pickle.dump(x_std, w_file)
pickle.dump(w, w_file)
w_file.close()

 
