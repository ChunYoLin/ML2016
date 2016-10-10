# -*- coding: utf8 -*-
import numpy as np
import re
import math
import sys
import json
import os
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

#  parse config file
cfg_data = json.load(open(sys.argv[2]))
feature = []
for item in cfg_data["feature"]:
    if item == 'A':
        feature = [v for k, v in hash_table.items()]
        break
    else:
        feature.append(hash_table[item])
feature.sort()
Regularization = cfg_data["Regularization"]
Scaling = cfg_data["Scaling"]
Learning_rate = cfg_data["Learning Rate"]
Optimizer = cfg_data["Optimizer"]
model = re.sub('.json', '', os.path.basename(sys.argv[2]))

weights_file_name = model + '_'
if Regularization > 0:
    weights_file_name += 'LAMBDA_' + str(Regularization) + '_'
if Scaling == True:
    weights_file_name += 'SC_'
weights_file_name += Optimizer + '_'
weights_file_name += '8_'


#  prepare x and y
train_data_arr = np.asarray(train_data, dtype=np.float32)
x = []
y = []
hour = 8
for i in range(len(train_data_arr[0]) - 9):
    x.append(train_data_arr[feature, i+9-hour:i+9])
    y.append(train_data_arr[hash_table["PM2.5"], i+9])
x = np.asarray(x, dtype = np.float32)
y = np.asarray(y, dtype = np.float32)
x = x.reshape(x.shape[0], hour * len(feature))

#  featrue scaling
if Scaling == True:
    x_mean = np.mean(x, axis = 0)
    x_std = np.std(x, axis = 0)
    x = (x - x_mean) / x_std 
bias = np.ones(shape = (x.shape[0], 1))
x = np.concatenate((bias, x), axis = 1)

#  model declaration
w = np.random.rand(x.shape[1])

#  train or validate model
data_set_size = x.shape[0]
train_set_size = data_set_size 
train_set = x[:train_set_size]
validate_set = x[train_set_size:]

#  train the model
if sys.argv[1] == "tr":
    t = 0
    iterations = 500000
    #  for adagrad
    if Optimizer == "Adagrad":
        gw_his = np.zeros(shape = (iterations, x.shape[1]))
    elif Optimizer == "Adam":
        Beta1 = 0.9
        Beta2 = 0.999
        e = 10**(-8)
        m = np.zeros(shape = w.shape)
        v = np.zeros(shape = w.shape)
    #  training
    LAMBDA = Regularization
    for k in range(iterations):
        L = 0
        t = t + 1
        y_ = np.dot(x[:train_set_size], w)
        L = np.sum((y[:train_set_size] - y_) ** 2) / (2 * train_set_size) + LAMBDA * np.sum(w**2) / 2
        gw = np.dot(-x[:train_set_size].transpose(), (y[:train_set_size] - y_)) / train_set_size
        if Optimizer == "NON":
            w -= Learning_rate * gw
        elif Optimizer == "Adam":
            m = Beta1 * m + (1 - Beta1) * gw
            v = Beta2 * v + (1 - Beta2) * (gw**2)
            m_ = m / (1 - Beta1**t)
            v_ = v / (1 - Beta2**t)
            w -= Learning_rate * m_ / (v_**(0.5) + e) * gw
        elif Optimizer == "Adagrad":
            gw_his[k] = (gw)**2
            w -= (Learning_rate / np.sum(gw_his, axis = 0)**0.5) * gw
        #  print out the current Loss and gradient of weights and bias
        gsum = np.sum(np.abs(gw)) 
        print "feature:", feature
        print "iter"+ str(k), "L:", L
        print "mean of gradient", gsum / (len(gw))
        print "Learning_rate:", Learning_rate
        print "Regularization:", LAMBDA
        print "Scaling:", Scaling
        print "Optimizer:", Optimizer
        if gsum / (len(gw)) < 0.00001:
            #  write the weights to file
            out = open('weights/' + weights_file_name + '.weights', 'w')
            out.write(str(k) + ' ')
            out.write(str(len(w)) + ' ')
            for i in range(len(w)):
                out.write(str(w[i]))
                out.write(' ')
            out.close()
            break
#  validate the model
elif sys.argv[1] == "vd":
    #  load model from weights file
    wf = open(sys.argv[3], 'r')
    wlist = wf.read().split(' ')
    w = np.asarray(wlist[2:(2 + int(wlist[1]))], dtype = np.float32)
    y_ = np.dot(x, w)
    e = np.abs(y - y_)
    e_train = np.sum(e[:train_set_size]) / train_set_size
    e_test = np.sum(e[train_set_size:]) / (data_set_size - train_set_size + 0.000000001)
    print "e_train = ", e_train, "e_test = ", e_test 
