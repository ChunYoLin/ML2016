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
cfg_data = json.load(open(sys.argv[1]))
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
model = re.sub('.json', '', os.path.basename(sys.argv[1]))

weights_file_name = model + '_'
weights_file_name += "NOR_"
#  weights_file_name += "SQUARE_"
if Scaling == True:
    weights_file_name += 'SC_'


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
#  x_2 = x**2
#  x = np.concatenate((x, x_2), axis = 1)
x = np.concatenate((bias, x), axis = 1)

#  train set and validate set
data_set_size = x.shape[0]
#  train_set_size = data_set_size 
train_set_size = 4000 
train_set = x[:train_set_size]
validate_set = x[train_set_size:]

#  caculate the model by normal equation
w = np.dot(np.dot(np.linalg.inv(np.dot(x[:train_set_size].transpose(), x[:train_set_size])), x[:train_set_size].transpose()), y[:train_set_size])

#  validate the model
y_ = np.dot(x, w)
e = np.abs(y - y_)
e_train = np.sum(e[:train_set_size]) / train_set_size
e_test = np.sum(e[train_set_size:]) / (data_set_size - train_set_size + 0.0000000001)
print "e_train = ", e_train, "e_test = ", e_test
#  write out the model
out = open('weights_kaggle/' + weights_file_name + '.weights', 'w')
out.write(str(0) + ' ')
out.write(str(len(w)) + ' ')
for i in range(len(w)):
    out.write(str(w[i]))
    out.write(' ')
out.close()

