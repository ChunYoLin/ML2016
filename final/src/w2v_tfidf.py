import sys
import re
import operator
import math
import csv
import numpy as np
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec
from gensim import corpora

print "Loading model"
W2V_MODEL = Word2Vec.load('w2vmodel')

STOP_WORDS = set()
COMPOUND_WORDS = set()
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def token_string(string):
	string = re.sub('<.*?>', ' ', string)
	string = re.sub('\$[^\$]*\$', ' ', string)
	string = re.sub('<code>[\s\S]*<\/code>', ' ', string)
	string = re.sub('http:[\/\.\w\-%]+', ' ', string)
	string = re.sub('&[\w]*', ' ', string)
	string = re.sub('\\\[\w]*', ' ', string)
	string = re.sub('{,*?}', ' ', string)
	str_list = re.sub('[^A-Za-z0-9\' ]',' ' ,string).split(' ')
	return [string.lower() for string in str_list \
				if bool(string) is True]

def store_word(_word, _dict):
	_word = re.sub('\'', '', _word)
	if _word not in _dict:
		_dict[_word] = 1
	else :
		_dict[_word] += 1
	
def store_dic(_list, _dict):
	for word in _list:
		store_word(word, _dict)

def doc_2_bow(_list, _cmp_dict, _one_dict):
	pre_word = ''
	pre_2_word = ''
	for word in _list:
		if is_number(word) is True \
			or word in STOP_WORDS:
			continue
		store_word(word, _one_dict)
		_2_word = pre_word + '-' + word
		_3_word = pre_2_word + '-' +word
		pre_word = word
		pre_2_word = _2_word
		if _2_word in COMPOUND_WORDS:
			store_word(_2_word, _cmp_dict)
		if (_2_word +'s') in COMPOUND_WORDS:
			store_word(_2_word+'s', _cmp_dict)
		if _3_word in COMPOUND_WORDS:
			store_word(_3_word, _cmp_dict)

def find_similar(_list, _cmp_dict):
	vec = np.zeros(100)
	num = 0
	for word in _list:
		if word not in STOP_WORDS and word in W2V_MODEL.vocab:
			vec += W2V_MODEL[word]
			num += 1
	if num > 0:
		vec /= num
	for u_word, value in W2V_MODEL.similar_by_vector(vec):
		word = str(u_word)
		if word in COMPOUND_WORDS:
			store_word(word, _cmp_dict)
def read_list(_file):
	with open(_file, 'rb') as F:
		_data = F.read()
	_list = re.split(', ', _data)
	return _list

file_name = sys.argv[1]
stop_file = "stopword.txt"
compound_file = "compound_word.txt"
remove_tags = set()
print "Reading Data"
STOP_LIST = read_list(stop_file)
[ STOP_WORDS.add( word ) for word in STOP_LIST ]
COMPOUND_LIST = read_list(compound_file)
[ COMPOUND_WORDS.add(word) for word in COMPOUND_LIST ]
in_file =  open(file_name, 'rb')
data = csv.DictReader(in_file)


print "Processing Data"

id_list = []
title_list = []
content_list = []
doc_list = []
doc_2w_list = []
word_dic = {}
i=0
for doc in data:
	titles = token_string(doc['title'])

	contents = token_string(doc['content'])
	title_dic = {}
	content_dic = {}
	doc_dic = {}
	doc_2w_dic = {}
	find_similar(titles, doc_2w_dic)
	find_similar(contents, doc_2w_dic)
	
	doc_2_bow(titles, doc_2w_dic, title_dic)
	doc_2_bow(contents, doc_2w_dic, doc_dic)

	store_dic(doc_dic, word_dic)
	store_dic(doc_2w_dic, word_dic)
	
	title_list.append(title_dic)
	doc_list.append(doc_dic)
	doc_2w_list.append(doc_2w_dic)
	id_list.append(int(doc['id']))
del data
in_file.close()
print "id len : ",len(id_list)
#print "content len : ",len(content_list)
#print "title len : ",len(title_list)
doc_num = len(id_list)
#'''
print "Calculating TF_IDF"
#'''
TFIDF_list = []
for i in range(doc_num):
	TFIDF_dic = {}
	for word, value in doc_list[i].iteritems():
		TFIDF = value * math.log(float(doc_num)/float(word_dic[word]))
		if word_dic[word] < 10 or len(word) <= 2:
			continue
		if word in title_list[i]:
			TFIDF *= 2
		TFIDF_dic[word] = TFIDF
	TFIDF_list.append(TFIDF_dic)	

#'''

#'''
tag_dic = {}
possible_tags = set()

print "Observe tags"
for i in range(doc_num):
	tags = sorted(TFIDF_list[i], key=TFIDF_list[i].get, reverse=True)[:5]
	for _2_word in doc_2w_list[i]:
		tags.append( _2_word )
	store_dic( tags, tag_dic )

FP_num = 0
TN_num = 0
TP_num = 0

sort_tag = sorted(tag_dic.items(), key=operator.itemgetter(1),\
		reverse=True)
for j in range(len(sort_tag)):
	if sort_tag[j][1] <= 100 :
		break		
	word =  sort_tag[j][0]
	tag = word
	ocur_num = word_dic[word]
	
	if (word+'s') in word_dic:
		words = word + 's'
		prob =  float(word_dic[words]) / float(ocur_num)
		
		if word_dic[words] > 224 and prob > 0.27:
			tag = words
		else:
			remove_tags.add(words)
	if (word+'es') in word_dic:
		wordes = word + 'es'
		prob = float(ocur_num) / float(word_dic[wordes])
		if word_dic[wordes] > 224 and prob > 0.27:
			tag = wordes
		else:
			remove_tags.add(wordes)
	if tag not in remove_tags:	
		possible_tags.add(tag)

out_file = open('sub_tfidf_final.csv', 'wb+')
output = csv.writer(out_file)
output.writerow(['id','tags'])
#'''
cmp_2_set = set()
for word in COMPOUND_WORDS :
	split_word = word.split('-')
	fir = split_word[0]
	sec = split_word[1]
	if fir not in cmp_2_set:
		cmp_2_set.add(fir)
	if sec not in cmp_2_set:
		cmp_2_set.add(sec)
#'''
print "Writing tags"
for i in range(doc_num):
	tags = sorted(TFIDF_list[i], key=TFIDF_list[i].get, reverse=True)[:5]
	answer = []
	for _2_word in doc_2w_list[i]:
		tags.append( _2_word )
	for tag in tags:
		tags = tag + 's'
		tages = tag + 'es'
		if tags in possible_tags and tags not in answer \
					and tags not in cmp_2_set:
			answer.append(tags)
		elif tag in possible_tags and tag not in answer \
					and tag not in cmp_2_set:
			answer.append(tag)
		elif tages in possible_tags and tages not in answer \
					and tages not in cmp_2_set:
			answer.append(tages)
		
	output.writerow( [ id_list[i], " ".join(sorted(answer)) ] )
	store_dic( tags, tag_dic )
out_file.close()

	
#'''



