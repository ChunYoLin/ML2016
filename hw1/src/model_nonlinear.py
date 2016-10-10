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
weights_file_name += 'NONLINEAR_'
if Regularization > 0:
    weights_file_name += 'LAMBDA_' + str(Regularization) + '_'
if Scaling == True:
    weights_file_name += 'SC_'
weights_file_name += Optimizer + '_'


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
x_2 = x**2
x = np.concatenate((x, x_2), axis = 1)

#  featrue scaling
if Scaling == True:
    x_mean = np.mean(x, axis = 0)
    x_std = np.std(x, axis = 0)
    print x[0]
    print x[1]
    x = (x - x_mean) / x_std 
    print x[0]
    print x[1]

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
    t = 0
    iterations = 50000
    #  for adagrad
    if Optimizer == "Adagrad":
        gw_his = np.zeros(shape = (iterations, x.shape[1]))
        gb_his = np.zeros(shape = (iterations))
    elif Optimizer == "Adam":
        Beta1 = 0.9
        Beta2 = 0.999
        e = 10**(-8)
        m = np.zeros(shape = w.shape)
        v = np.zeros(shape = w.shape)
        mb = 0. 
        vb = 0. 
    #  training
    LAMBDA = Regularization
    for k in range(iterations):
        L = 0
        gw = np.zeros(shape = w.shape)
        gb = 0
        t = t + 1
        for i, data in enumerate(train_set):
            wx = np.dot(w.transpose(), data)
            y_ = b + wx
            L += (1 / 2.) * (y[i] - y_)**2 / train_set_size 
            for j in range(len(data)):
                gw[j] += ((y[i] - y_) * (-data[j]) + LAMBDA * w[j]) / train_set_size
            gb += (y[i] - y_) / train_set_size
        L += LAMBDA * np.sum(w**2) / 2
        if Optimizer == "NON":
            w -= Learning_rate * gw
            b -= Learning_rate * gb 
        elif Optimizer == "Adam":
            m = Beta1 * m + (1 - Beta1) * gw
            v = Beta2 * v + (1 - Beta2) * (gw**2)
            m_ = m / (1 - Beta1**t)
            v_ = v / (1 - Beta2**t)
            mb = Beta1 * mb + (1 - Beta1) * gb
            vb = Beta2 * vb + (1 - Beta2) * (gb**2)
            mb_ = mb / (1 - Beta1**t)
            vb_ = vb / (1 - Beta2**t)
            w -= Learning_rate * m_ / (v_**(0.5) + e) * gw
            b -= Learning_rate * mb_ / (vb_**(0.5) + e) * gb
        elif Optimizer == "Adagrad":
            gw_his[k] = (gw)**2
            gb_his[k] = (gb)**2
            w -= (Learning_rate / np.sum(gw_his, axis = 0)**0.5) * gw
            b -= (Learning_rate / np.sum(gb_his, axis = 0)**0.5) * gb 
        #  print out the current Loss and gradient of weights and bias
        gsum = np.sum(np.abs(gw)) + gb
        print "feature:", feature
        print "iter"+ str(k), "L:", L
        print "gradient of w,b :", gw, gb
        print "mean of gradient", gsum / (len(gw) + 1)  
        print "Learning_rate:", Learning_rate
        print "Regularization:", LAMBDA
        print "Scaling:", Scaling
        print "Optimizer:", Optimizer
        if gsum / (len(gw) + 1) < 0.1:
            #  write the weights and bias to file
            out = open('weights/' + weights_file_name + '.weights', 'w')
            out.write(str(k + 1 + time) + ' ')
            out.write(str(len(w)) + ' ')
            for i in range(len(w)):
                out.write(str(w[i]))
                out.write(' ')
            out.write(str(b))
            out.write(' ')
            out.write(str(L))
            out.close()
            break
#  validate the model
elif sys.argv[1] == "vd":
    e_train = 0
    e_test = 0
    y_ = np.zeros(shape = y.shape)
    for i, data in enumerate(x):
        wx = np.dot(w.transpose(), data)
        y_[i] = b + wx
        if i < train_set_size:
            e_train += np.abs(y[i] - y_[i])
        else:
            e_test += np.abs(y[i] - y_[i])
    print "e_train = ", e_train / train_set_size, "e_test = ", e_test / (data_set_size - train_set_size)
