# -*- coding: utf8 -*-
import numpy as np
import re
import math

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
        train_data[hash_table[data[2]]]+=data_float

#  prepare x and y
train_data_arr = np.asarray(train_data, dtype=np.float32)
x = []
y = []
for i in range(len(train_data_arr[0])-9):
    x.append(train_data_arr[:,i:i+9])
    y.append(train_data_arr[hash_table["PM2.5"],i+9])
x = np.asarray(x, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

#  model declartion
w = np.random.rand(x.shape[1])
b = np.random.rand()

#  load weights from weights file
wf = open('./weights', 'r')
wlist = wf.read().split(' ')
w = np.asarray(wlist[1:int(wlist[0])+1], dtype = np.float32)
b = float(wlist[int(wlist[0])+1])

#  inference
train_size = 2000
e_train = 0
e_test = 0
y_ = np.zeros(shape = y.shape)
for i ,data in enumerate(x[:train_size]):
    wx = np.dot(w.transpose(), x[i])
    y_[i] = b + wx
    e_train += np.abs(y[i] - y_[i])
    
for i, data in enumerate(x[train_size:]):
    wx = np.dot(w.transpose(), x[i])
    y_[i] = b + wx
    e_test += np.abs(y[i] - y_[i])
print e_train,e_test 
