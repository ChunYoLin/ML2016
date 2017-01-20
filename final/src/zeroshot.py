import pandas as pd
import numpy as np
import re
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec
from gensim import corpora

def clean_string(string):
    string = re.sub('<.*?>', ' ', string)
    string = re.sub('http:[\/\.\w\-%]+', ' ', string)
    string = re.sub('&[\w]*', ' ', string)
    string = re.sub('\$[^\$]*\$', ' ', string)
    return string

stopwords = set()
with open('./data/stopwordv2.txt') as f:
    for w in f.read().split(', '):
        stopwords.add(w)

def get_w2vmodel(corpus):
    model = Word2Vec(size = 100, window = 8, min_count = 1, workers = 4)
    model.build_vocab(corpus)
    for i in range(5):
        model.train(corpus)
    return model

def get_docvec(corpus, w2vmodel):
    doc_vec = []
    for content in corpus:
        vec = np.zeros(100)
        num = 0
        for word in content:
            if word not in stopwords and word in w2vmodel.vocab:
                vec += w2vmodel[word]
                num += 1
        vec /= num
        doc_vec.append(vec)
    doc_vec = np.asarray(doc_vec)
    return doc_vec

def get_tagvec(tag_list, w2vmodel):
    tag_vec = []
    for idx, tag in enumerate(tag_list):
        t_vec = np.zeros(100)
        num = 0
        for t in tag.split():
            if t in w2vmodel.vocab:
                t_vec += w2vmodel[t]
                num += 1
        if num > 0:
            t_vec /= num
        tag_vec.append(t_vec)
    tag_vec = np.asarray(tag_vec)
    return tag_vec

class CORPUS:
    def __init__(self, fpath):
        corpus = pd.read_csv(fpath)
        corpus['content'] = corpus['content'].apply(lambda x: clean_string(x))
        corpus['content'] = corpus['content'].apply(lambda x: re.split('\W+', x.lower()))
        corpus['title'] = corpus['title'].apply(lambda x: clean_string(x))
        corpus['title'] = corpus['title'].apply(lambda x: re.split('\W+', x.lower()))
        corpus['ti_cont'] = corpus['title'] + corpus['content']
        self.corpus = corpus
        cor_dict = corpora.Dictionary(corpus['content'])
        cor = [cor_dict.doc2bow(text) for text in corpus['content']]
        self.cor_dict = cor_dict
        self.cor = cor
        tfidf = TfidfModel(cor)
        self.tfidf = tfidf
        def idf(self, W):
            return self.tfidf.idfs.get(self.cor_dic.token2id[W])

train0 = CORPUS('./data/cooking.csv')
train1 = CORPUS('./data/travel.csv')
train2 = CORPUS('./data/biology.csv')
train3 = CORPUS('./data/diy.csv')
train4 = CORPUS('./data/crypto.csv')
valid = CORPUS('./data/robotics.csv')

corpus_all = []
corpus_train = []
for doc in train0.corpus['ti_cont']:
    corpus_train.append(doc)
    corpus_all.append(doc)
for doc in train1.corpus['ti_cont']:
    corpus_train.append(doc)
    corpus_all.append(doc)
for doc in train2.corpus['ti_cont']:
    corpus_train.append(doc)
    corpus_all.append(doc)
for doc in train3.corpus['ti_cont']:
    corpus_train.append(doc)
    corpus_all.append(doc)
for doc in train4.corpus['ti_cont']:
    corpus_train.append(doc)
    corpus_all.append(doc)
for doc in valid.corpus['ti_cont']:
    corpus_all.append(doc)


tag_all = []
for tag in train0.corpus['tags']:
    tag_all.append(tag)
for tag in train1.corpus['tags']:
    tag_all.append(tag)
for tag in train2.corpus['tags']:
    tag_all.append(tag)
for tag in train3.corpus['tags']:
    tag_all.append(tag)
for tag in train4.corpus['tags']:
    tag_all.append(tag)
#  #  for doc in valid.corpus['content']:
    #  #  corpus_all.append(doc)
all_model = get_w2vmodel(corpus_all)
train_docvec = get_docvec(corpus_train, all_model)
train_tagvec = get_tagvec(tag_all, all_model)
valid_docvec = get_docvec(valid.corpus['ti_cont'], all_model)
valid_tagvec = get_tagvec(valid.corpus['tags'], all_model)
print train_docvec.shape, train_tagvec.shape, valid_docvec.shape, valid_tagvec.shape

#  #  train_model = get_w2vmodel(train.corpus['content'])
#  #  train_docvec = get_docvec(train.corpus['content'], train_model)
#  #  train_tagvec = get_tagvec(train.corpus['tags'], train_model)

#  #  valid_model = get_w2vmodel(valid.corpus['content'])
#  #  valid_docvec = get_docvec(valid.corpus['content'], valid_model)
#  #  valid_tagvec = get_tagvec(valid.corpus['tags'], valid_model)
        
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import tensorflow as tf
tf.python.control_flow_ops = tf

d2tmodel = Sequential()
d2tmodel.add(Dense(5000, input_dim = 100))
d2tmodel.add(Activation('relu'))
d2tmodel.add(Dense(2500))
d2tmodel.add(Activation('relu'))
d2tmodel.add(Dense(1000))
d2tmodel.add(Activation('relu'))
d2tmodel.add(Dense(100))
#  d2tmodel.add(Activation('linear'))
d2tmodel.compile(loss = 'mse', optimizer = 'adam')
d2tmodel.fit(train_docvec, train_tagvec, nb_epoch = 10, batch_size = 200, shuffle = True, validation_data = (valid_docvec, valid_tagvec))
train_pred = d2tmodel.predict(train_docvec)
valid_pred = d2tmodel.predict(valid_docvec)
with open('train_pred', 'w') as f:
    for t_v in train_pred:
        for word in all_model.similar_by_vector(t_v)[:5]:
            f.write(word[0] + ' ')
        f.write('\n')
with open('valid_pred', 'w') as f:
    for t_v in valid_pred:
        for word in all_model.similar_by_vector(t_v)[:5]:
            f.write(word[0] + ' ')
        f.write('\n')

#  with open('out.txt', 'w') as out:
    #  out.write('id,tags\n')
    #  for idx, w_l in enumerate(sim):
        #  out.write(str(corpus['id'][idx]) + ',')
        #  for w in w_l:
            #  out.write(str(w) + ' ')
        #  out.write('\n')
