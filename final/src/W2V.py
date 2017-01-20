import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

print "Read Sentences"
sentences = LineSentence('clean_text.txt')
model = Word2Vec(size = 100, window = 8, min_count = 1, workers = 4)
print "Training Model"
model.build_vocab(sentences)
for i in range(5):
	model.train(sentences)
print "Saving Model"
model.save('w2vmodel')

