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

#  select feature
feature = []
fea_file = open(sys.argv[1])
s = fea_file.read().rstrip('\n')
if s == 'A':
    feature = [v for k, v in hash_table.items()]
else:
    for fea in s.split(','):
        feature.append(hash_table[fea])
feature.sort()

#  parse data
test_data = np.zeros(shape = (240,18,9))
test_file = open('./data/test_X.csv', 'r')
for idx, row in enumerate(test_file):
    data = re.sub('\n|\r', '', row).split(',')
    for idx, v in enumerate(data[2:]):
        if v != 'NR':
            test_data[int(data[0][3:]), hash_table[data[1]], idx] = v
        else:
            test_data[int(data[0][3:]), hash_table[data[1]], idx] = 0 

x = test_data[:,feature]
test_set_size = x.shape[0]
x = x.reshape(test_set_size, 9 * len(feature))
y = np.zeros(shape = test_set_size)

#  load model from weights file
wf = open(sys.argv[2], 'r')
wlist = wf.read().split(' ')
w = np.asarray(wlist[2:(2 + int(wlist[1]))], dtype = np.float32)
b = float(wlist[int(wlist[1]) + 1])

#  predict the data
for i, data in enumerate(x):
    wx = np.dot(w.transpose(), x[i])
    y[i] = wx + b

#  output
out = open('pred_' + str(len(feature)) + '.csv', 'w')
out.write("id,value\n")
for i in range(y.shape[0]):
    out.write('id_' + str(i) + ',' + str(y[i]) + '\n')    
out.close()
