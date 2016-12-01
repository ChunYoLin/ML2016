from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import re
import matplotlib.pyplot as plt
from preprocess import Corpus

title = Corpus(c = 'title')
fea = title.bow * np.log(len(title.corpus) / title.df)
fea = title.bow
df = np.zeros(shape = len(title.word_set))
for k, v in title.word_set.iteritems():
    df[title.word_set_id[k]] = title.word_set_total[k]
index = np.argsort(df)[::-1][:1000]
fea = fea[:, index]
rank = 0
for i in index:
    for k, v in title.word_set_id.iteritems():
        if v == i:
            rank += 1
            print 'top %d word %s'%(rank, k)

center = []
center_arg = []
center.append(np.full((fea.shape[1]), 0., dtype = np.float32))
while(True):
    index_arr = np.arange(len(title.corpus))
    np.random.shuffle(index_arr)
    index_lst = []
    find = 0
    for i in index_arr:
        if fea[i].any() != 0.:
            index_lst.append(i)
        if len(index_lst) == 20:
            mat = np.matmul(fea[index_lst], fea[index_lst].transpose())
            np.fill_diagonal(mat, 0.)
            if mat.any() == 0.:
                for idx in index_lst:
                    center.append(fea[idx])
                find = 1
                break
            index_lst = []
    if find == 1:
        break

center = np.asarray(center)
Group = KMeans(n_clusters = 21, init = center, n_init = 1, random_state = 0).fit_predict(fea)
tag = np.zeros(shape = 21, dtype = np.int32)
for c in Group:
    tag[c] += 1
print len(title.word_set)
print tag
with open('./check_index.csv', 'r') as f_in, open('pred.csv', 'w') as f_out:
    f_out.write('ID,Ans\n')
    for idx, pair in enumerate(f_in):
        p = pair.split(',')
        if idx > 0:
            if Group[int(p[1])] == Group[int(p[2])]:
                f_out.write(str(p[0]) + ',' + str(1) + '\n')
            else:
                f_out.write(str(p[0]) + ',' + str(0) + '\n')

reduced_data = PCA(n_components = 2).fit_transform(fea)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c = Group * 5, s = 20)
plt.show()
