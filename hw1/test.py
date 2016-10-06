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

#  parse data
x = np.zeros(shape = (240,18,9))
test_data = open('./data/test_X.csv', 'r')
for idx, row in enumerate(test_data):
    data = re.sub('\n|\r', '', row).split(',')
    for idx, v in enumerate(data[2:]):
        if v != 'NR':
            x[int(data[0][3:]), hash_table[data[1]], idx] = v
        else:
            x[int(data[0][3:]), hash_table[data[1]], idx] = 0 

#  x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_ = x[:,hash_table["PM2.5"]] 
y = np.zeros(shape = x.shape[0])

#  model declartion
w = np.random.rand(x.shape[1])
b = np.random.rand()

#  load weights from weights file
wf = open('./weights', 'r')
wlist = wf.read().split(' ')
w = np.asarray(wlist[1:int(wlist[0])+1], dtype = np.float32)
b = float(wlist[int(wlist[0])+1])

#  predict the data
for i, data in enumerate(x):
    wx = np.dot(w.transpose(), x_[i])
    y[i] = wx + b

#  output
out = open('./pred.csv','w')
out.write("id,value\n")
for i in range(y.shape[0]):
    out.write('id_' + str(i) + ',' + str(y[i]) + '\n')    
out.close()
