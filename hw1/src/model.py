# -*- coding: utf8 -*-
import numpy as np
import re
import math
import sys

#  hash_table for key
hash_table = {
    "AMB_TEMP":0, "CH4":1, "CO":2, "NMHC":3, "NO":4, "NO2":5, 
    "NOx":6, "O3":7, "PM10":8, "PM2.5":9, "RAINFALL":10, 
    "RH":11, "SO2":12, "THC":13, "WD_HR":14, "WIND_DIREC":15,
    "WIND_SPEED":16, "WS_HR":17
}
inv_hash_table = {v: k for k, v in hash_table.items()}

#  prepare data
train_data = [[] for i in range(18)]
raw_file = open('./data/train.csv')
for idx,row in enumerate(raw_file):
    if idx > 0:
        data = re.sub('\r|\n', '', row).split(',')
        data_float = []
        for x in data[3:]:
            if x != 'NR':
                data_float.append(x)
            else:
                data_float.append(0)
        train_data[hash_table[data[2]]] += data_float

#  select feature
feature = []
fea_file = open(sys.argv[2])
s = fea_file.read().rstrip('\n')
if s == 'A':
    feature = [v for k, v in hash_table.items()]
else:
    for fea in s.split(','):
        feature.append(hash_table[fea])
feature.sort()

#  prepare x and y
train_data_arr = np.asarray(train_data, dtype=np.float32)
x = []
y = []
for i in range(len(train_data_arr[0]) - 9):
    x.append(train_data_arr[feature, i:i+9])
    y.append(train_data_arr[hash_table["PM2.5"], i+9])
x = np.asarray(x, dtype = np.float32)
y = np.asarray(y, dtype = np.float32)
x = x.reshape(x.shape[0], 9 * len(feature))
x_mean = np.mean(x, axis = 0)
x_std = np.std(x, axis = 0)

#  model declaration
time = 0
if sys.argv[3] == "NW":
    #  initial weights
    w = np.random.rand(x.shape[1])
    b = np.random.rand() 
else:
    #  load weights from weights file
    wf = open(sys.argv[3], 'r')
    wlist = wf.read().split(' ')
    time = int(wlist[0])
    w = np.asarray(wlist[2:(2 + int(wlist[1]))], dtype = np.float32)
    b = float(wlist[int(wlist[1]) + 1])

#  train or validate model
data_set_size = x.shape[0]
train_set_size = 4000
train_set = x[:train_set_size]
validate_set = x[train_set_size:]

#  train the model
if sys.argv[1] == "tr":
    #  for adagram
    iterations = 50000
    gw_his = np.zeros(shape = (iterations, x.shape[1]))
    gb_his = np.zeros(shape = (iterations))

    #  training
    learning_rate = float(sys.argv[4])
    LAMBDA = 100
    for k in range(iterations):
        L = 0
        gw = np.zeros(shape = w.shape)
        gb = 0
        for i, data in enumerate(train_set):
            wx = np.dot(w.transpose(), data)
            y_ = b + wx
            L += (1 / (2. * train_set_size)) * (y[i] - y_)**2 
            for j in range(len(data)):
                gw[j] += ((y[i] - y_) * (-data[j]) + LAMBDA * w[j]) / train_set_size
            gb += (y[i] - y_) / train_set_size
        gw_his[k] = (gw)**2
        gb_his[k] = (gb)**2
        #  exit when gw and gb extremely small
        if np.sum(np.abs(gw)) + gb < 0.0001:
            #  write the final model to file
            out = open('weights/' + str(len(feature)) + '_' + 'final' + '.weights', 'w')
            out.write(str(k) + ' ')
            out.write(str(len(w)) + ' ')
            for i in range(len(w)):
                out.write(str(w[i]))
                out.write(' ')
            out.write(str(b))
            out.close()
        L += LAMBDA * np.sum(w**2) / 2
        w -= (learning_rate / np.sum(gw_his, axis = 0)**0.5) * gw
        b -= (learning_rate / np.sum(gb_his, axis = 0)**0.5) * gb 
        #  print out the current Loss and gradient of weights and bias
        print "iter"+str(k),"L:", L, gw, gb
        #  if iter 5000 times write the current weights and bias to file
        if (k + 1) % 5000 == 0 and k != 0:
            out = open('weights/' + str(len(feature)) + '_' + str((k + 1) + time) + '.weights', 'w')
            out.write(str(k) + ' ')
            out.write(str(len(w)) + ' ')
            for i in range(len(w)):
                out.write(str(w[i]))
                out.write(' ')
            out.write(str(b))
            out.close()
#  validate the model
elif sys.argv[1] == "vd":
    e_train = 0
    e_test = 0
    y_ = np.zeros(shape = y.shape)
    for i ,data in enumerate(x[:train_set_size]):
        wx = np.dot(w.transpose(), data)
        y_[i] = b + wx
        e_train += np.abs(y[i] - y_[i])
        
    for i, data in enumerate(x[train_set_size:]):
        wx = np.dot(w.transpose(), data)
        y_[i] = b + wx
        e_test += np.abs(y[i] - y_[i])
    print "e_train = ", e_train / train_set_size, "e_test = ", e_test / (data_set_size - train_set_size)
