import numpy as np
import tensorflow as tf
import input_data
import scipy.spatial.distance
import cPickle as pk

#  label_code = sess.run(encoder_op, feed_dict = {X: labeled_image})
# Network Parameters
n_hidden_1 = 8192 # 1st layer num features
n_hidden_2 = 4096 # 2nd layer num features
n_hidden_3 = 2048 # 3nd layer num features
n_hidden_4 = 1024 # 4nd layer num features
n_hidden_5 = 512 # 5nd layer num features
n_hidden_6 = 256 # 5nd layer num features
n_input = 3072

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'encoder_h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'encoder_h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_5])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_4])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h5': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h6': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'encoder_b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b5': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b6': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),biases['encoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['encoder_h5']),biases['encoder_b5']))
    layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['encoder_h6']),biases['encoder_b6']))
    return layer_6


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),biases['decoder_b5']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, weights['decoder_h6']),biases['decoder_b6']))
    return layer_6

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)

#  preprocessing data
dataset = input_data.CIFAR10()
labeled_image, label = dataset.labeled_image()
unlabeled_image = dataset.unlabeled_image()
all_image = np.concatenate((labeled_image, unlabeled_image), axis = 0) / 255.
batch_size = 200
batch = input_data.minibatch(all_image, batch_size = batch_size)
labeled_image /= 255.
unlabeled_image /= 255.

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)
e = 20
for epoch in range(e):
    for i in range(batch.shape[0]):
        _, c, y_p, y_t = sess.run([optimizer, cost, y_pred, y_true], feed_dict = {X: all_image[batch[i]]})
        print 'epoch ' + str(epoch + 1) + '/20' + ' batch '  + str(i + 1) + '/' + str(batch.shape[0]) +' cost:' + str(c) 


#---validate---#
train_image = []
train_label = []
validate_image = []
validate_label = []
train_set_size = 400
for i in range(10):
    for j in range(500):
        if j < train_set_size:
            train_image.append(labeled_image[i * 500 + j])
            train_label.append(label[i * 500 + j])
        else:
            validate_image.append(labeled_image[i * 500 + j])
            validate_label.append(label[i * 500 + j])
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
validate_image = np.asarray(validate_image)
validate_label = np.asarray(validate_label)
train_code = sess.run(encoder_op, feed_dict = {X: train_image})
validate_code = sess.run(encoder_op, feed_dict = {X: validate_image})
train_code = train_code.reshape(4000, -1)
validate_code = validate_code.reshape(1000, -1)
label_code = []
for i in range(0, train_set_size * 10, train_set_size):
    label_code.append(np.mean(train_code[i : i + train_set_size], axis = 0))
label_code = np.asarray(label_code)
self_label = np.ndarray(shape = (validate_code.shape[0], 10))
#  simularity = np.argmin(scipy.spatial.distance.cdist(validate_code, train_code, 'euclidean'), axis = 1)
simularity = scipy.spatial.distance.cdist(validate_code, label_code, 'cosine')
acc = 0.
for i in range(1000):
    print "image %d, label %d, code_label %d"%(i, np.argmax(validate_label[i]), np.argmin(simularity[i]))
    if np.argmax(validate_label[i]) == np.argmin(simularity[i]):
        acc += 1.
    print acc / 1000.

#  unlabel_code = []
#  for i in range(100):
    #  code = sess.run(encoder_op, feed_dict = {X: unlabeled_image[i * 450 : (i + 1) * 450]})
    #  unlabel_code.append(code)

#  unlabel_code = np.asarray(unlabel_code).reshape(45000, -1)
#  label_code = label_code.reshape(5000, -1)
#  print label_code.shape, unlabel_code.shape

#  #---label unlabel_image by encode_code cosine simiularity---#
#  self_label = np.ndarray(shape = (unlabel_code.shape[0], 10))
#  unlabel_simularity = np.argmin(scipy.spatial.distance.cdist(unlabel_code, label_code, 'cosine'), axis = 1)
#  simularity = scipy.spatial.distance.cdist(unlabel_code, label_code, 'cosine')
#  for i in range(self_label.shape[0]):
    #  print simularity[i]
    #  self_label[i] = label[unlabel_simularity[i]]

#  labeled_image = np.concatenate((labeled_image, unlabeled_image), axis = 0)
#  label = np.concatenate((label, self_label), axis = 0)
#  print 'dump the image and label......'

#  with open('image.pk', 'w') as f:
    #  pk.dump(labeled_image, f)
#  with open('label.pk', 'w') as f:
    #  pk.dump(label, f)


