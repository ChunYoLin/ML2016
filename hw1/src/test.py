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

#  parse config file
f = open(sys.argv[1], 'r')
cfg_data = json.load(f)
feature = []
for item in cfg_data["feature"]:
    if item == 'A':
        feature = [v for k, v in hash_table.items()]
        break
    else:
        feature.append(hash_table[item])
feature.sort()
Scaling = cfg_data["Scaling"]
Root = cfg_data["Square Root"]
Square = cfg_data["Square"]
Cubed = cfg_data["Cubed"]
Hour = cfg_data["Hour"]
model = re.sub('.json', '', os.path.basename(sys.argv[1]))


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

#  determine the model
x = test_data[:,feature,9-Hour:]
test_set_size = x.shape[0]
x = x.reshape(test_set_size, Hour * len(feature))
y = np.zeros(shape = test_set_size)
bias = np.ones(shape = (x.shape[0], 1))
x_root = (x + 10)**0.5
x_2 = x**2
x_3 = x**3
if Root == True:
    x = np.concatenate((x, x_root), axis = 1)
if Square == True:
    x = np.concatenate((x, x_2), axis = 1)
if Cubed == True:
    x = np.concatenate((x, x_3), axis = 1)
x = np.concatenate((bias, x), axis = 1)

#  load model from weights file
wf = open(sys.argv[2], 'r')
wlist = wf.read().split(' ')
w = np.asarray(wlist[2:(2 + int(wlist[1]))], dtype = np.float32)

#  predict the data
y = np.dot(x, w)

#  output
out_name = sys.argv[3]
print "writing predict results to file:", out_name 
out = open(out_name, 'w')
out.write("id,value\n")
for i in range(y.shape[0]):
    out.write('id_' + str(i) + ',' + str(y[i]) + '\n')    
out.close()
