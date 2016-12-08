import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ipdb
label = np.zeros(shape = (20000, 20))
color = np.zeros(shape = (20000, 1))
with open('./data/label_StackOverflow.txt') as f:
    for idx, row in enumerate(f.read().splitlines()):
        label[idx, int(row) - 1] = 1
        color[idx] = int(row)
reduced_data = TSNE(n_components = 2).fit_transform(label)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = color * 5, s = 20)
plt.show()
ipdb.set_trace()
