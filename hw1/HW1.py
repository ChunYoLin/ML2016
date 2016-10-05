# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import re
from datetime import datetime, date
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

#  prepare x
train_data_arr = np.asarray(train_data, dtype=np.float32)
x = [[] for i in range(18)]
for idx, item in enumerate(train_data_arr):
    for i in range(len(item)-9):
        x[idx].append(item[i:i+9])
x = np.asarray(x, dtype=np.float32)

#  prepare y
y = []
PM2_5 = train_data_arr[hash_table["PM2.5"]]
for i in range(len(PM2_5)-9):
    y.append(PM2_5[i+9])
y = np.asarray(y, dtype=np.float32)

# model declaration
trainset_size = len(y)
w = np.random.rand(trainset_size,18,9)
b = np.random.rand(trainset_size)


