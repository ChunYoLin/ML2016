from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import initializations
import numpy as np
import re
from keras import backend as K
def my_init(shape, name=None):
    #  value = np.random.random(shape)
    #  return K.variable(value, name=name)
    return initializations.normal(shape, scale=0.01, name=name)
train_data = open('./spam_data/spam_train.csv', 'r')
x = []
y = []
for row in train_data:
    row_l = re.sub('\n|\r', '', row).split(',')
    x.append(row_l[1 : 1 + 56])
    y.append(row_l[len(row_l) - 1])
x = np.asarray(x, dtype = np.float32)
bias = np.ones(shape = (x.shape[0], 1))
x = np.concatenate((bias, x), axis = 1)
w = np.zeros(shape = x.shape[1])
y = np.asarray(y, dtype = np.float32)
m = x.shape[0]
train_set_size = m

model = Sequential()
model.add(Dense(output_dim = 38, input_dim = x.shape[1], activation = 'sigmoid', init = my_init))
#  model.add(Dense(output_dim = 128, input_dim = 256, activation = 'sigmoid'))
model.add(Dense(output_dim = 1, input_dim = 38, activation = 'sigmoid', init = my_init))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x[:train_set_size], y[:train_set_size], nb_epoch = 20000, batch_size = 4001)

#  test_data = open('./spam_data/spam_test.csv', 'r')
#  x_ = []
#  for row in test_data:
    #  row_l = re.sub('\n|\r', '', row).split(',')
    #  x_.append(row_l[1 : 1 + 57])
#  x_ = np.asarray(x_, dtype = np.float32)
#  bias = np.ones(shape = (x_.shape[0], 1))
#  x_ = np.concatenate((bias, x_), axis = 1)
#  y_ = model.predict(x_)
print model.evaluate(x[train_set_size:], y[train_set_size:])
print model.summary()

#  pred = open('./pred.csv', 'w')
#  pred.write('id,label\n')
#  for i in range(y_.shape[0]):
    #  if y_[i] > 0.5:
        #  pred.write(str(i + 1) + ',' + str(1) + '\n')
        #  y_[i] = 1
    #  else:
        #  pred.write(str(i + 1) + ',' + str(0) + '\n')
        #  y_[i] = 0
#  pred.close()
