import sys
import re
import numpy as np
import nltk
from collections import OrderedDict
class Corpus:
    def __init__(self, c = 'both', file_path = './data/'):
        #  corpus
        corpus = []
        if c == 'title':
            with open(file_path + 'title_StackOverflow.txt') as f:
                for row in f:
                    corpus.append(re.split('\W+', row.lower()))
        elif c == 'doc':
            with open(file_path + 'docs.txt') as f:
                for row in f:
                    corpus.append(re.split('\W+', row.lower()))
        elif c == 'both':
            with open(file_path + 'title_StackOverflow.txt') as f1, open(file_path + 'docs.txt') as f2:
                for row in f1:
                    corpus.append(re.split('\W+', row.lower()))
                for row in f2:
                    corpus.append(re.split('\W+', row.lower()))
        self.corpus = corpus
        #  word set
        word_set = {}
        word_set_total = {}
        for doc in corpus:
            df_list = []
            for word in doc:
                word = word.lower()
                if word not in word_set:
                    word_set[word] = 1
                    word_set_total[word] = 1
                    df_list.append(word)
                elif word not in df_list:
                    word_set[word] += 1
                    df_list.append(word)
                else:
                    word_set_total[word] += 1

        self.word_set = word_set
        self.word_set_total = word_set_total
        #  stop word set
        word_set_sorted = OrderedDict(sorted(self.word_set.items(), key = lambda x: x[1], reverse = True))
        ig_word = set()
        stopwords = nltk.corpus.stopwords.words('english')
        for w in stopwords:
            ig_word.add(w)
        IDX = 0
        for k, v in word_set_sorted.iteritems():
            IDX += 1
            if c == 'both':
                if IDX <= 20:
                    ig_word.add(k)
            if v < 4:
                ig_word.add(k)
            if len(k) < 2:
                ig_word.add(k)
        self.ig_word = ig_word
        #  remove word in ig set from wd set
        for k in ig_word:
            if k in self.word_set:
                del self.word_set[k]
        #  word id
        word_set_id = {}
        w_id = 0
        for k in self.word_set.keys():
            word_set_id[k] = w_id
            w_id += 1
        self.word_set_id = word_set_id
        #  bow
        if c == 'title':
            bow = np.zeros(shape = (len(self.corpus), len(self.word_set)))
            for idx, t in enumerate(self.corpus):
                for w in t:
                    if w not in self.ig_word:
                        bow[idx, self.word_set_id[w]] += 1
            self.bow = bow
        #  doc_freq
        df = np.zeros(shape = len(self.word_set))
        for k, v in self.word_set.iteritems():
            df[self.word_set_id[k]] = self.word_set[k]
        self.df = df
        
            

            
