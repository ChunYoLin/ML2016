import cPickle as pk
import numpy as np

def minibatch(X, batch_size = 50, Shuffle = True):
    all_batch = np.arange(X.shape[0])
    all_size = X.shape[0]
    if Shuffle:
        np.random.shuffle(all_batch)
    batch = []
    for a in range(all_size / batch_size):
        single_batch = []
        for b in range(batch_size):
            single_batch.append(all_batch[a * batch_size + b])
        batch.append(single_batch)
    batch = np.asarray(batch)
    return batch

class CIFAR10:
    def __init__(self):
        pass
    def labeled_image(self):
        #  loading data from pikcle file
        all_label = pk.load(open('./data/all_label.p', 'rb'))
        all_label = np.asarray(all_label, dtype = np.float32)
        labeled_image = all_label.reshape(all_label.shape[0] * all_label.shape[1], all_label.shape[2])
        label = np.zeros(shape = (labeled_image.shape[0], 10), dtype = np.float32)
        for i in range(10):
            label[i * 500 : i * 500 + 500, i] = 1
        return labeled_image, label
    def unlabeled_image(self):
        #  loading data from pikcle file
        all_unlabel = pk.load(open('./data/all_unlabel.p', 'rb'))
        unlabeled_image = np.asarray(all_unlabel, dtype = np.float32)
        return unlabeled_image
    def test_image(self):
        #  loading data from pikcle file
        test = pk.load(open('./data/test.p', 'rb'))
        test_image = np.asarray(test['data'], dtype = np.float32)
        return test_image
