# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import re
from datetime import datetime, date
columns = ["date", "station", "item"] + [str(i) for i in range(24)]
raw_data = pd.read_csv('./data/train.csv', names = columns)

raw_data = raw_data.drop("station", 1).drop(raw_data.index[0]).set_index(["date"])
train_data = pd.DataFrame()
train_file = open('./data/train.csv')

for idx, line in enumerate(train_file):
    if idx > 0:
        raw_list = re.sub('\r\n', '', line).split(',')
        date = datetime.strptime(raw_list[0], "%Y/%m/%d")
        item = raw_list[2]
        data = raw_list[3:]
        train_data[date, item] = data
print train_data[datetime(2014,1,1), "PM2.5"][23]
