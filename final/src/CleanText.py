import sys
import re
import operator
import math
import pandas as pd
import csv

COMPOUND_WORDS = set()
def token_string(string):
	string = re.sub('<.*?>', '\n', string)
	string = re.sub('\$[^\$]*\$', ' ', string)
	string = re.sub('<code>[\s\S]*<\/code>', ' ', string)
	string = re.sub('http:[\/\.\w\-%]+', ' ', string)
	string = re.sub('&[\w]+', ' ', string)
	string = re.sub('\\\[\w]+', ' ', string)
	string = re.sub('{,*?}', ' ', string)

	str_list = re.sub('[^A-Za-z0-9\n ]','' ,string).split(' ')
	return [string.lower() for string in str_list \
				if bool(string) is True]

def doc2cmp(_list, _cmp_list):
	pre_word = ''
	pre_2_word = ''
	for word in _list:
		_cmp_list.append(word)
		_2_word = pre_word + '-' + word
		_3_word = pre_2_word + '-' +word
		pre_word = word
		pre_2_word = _2_word
		if _2_word in COMPOUND_WORDS:
			_cmp_list.pop()
			_cmp_list.pop()
			_cmp_list.append(_2_word)
		if (_2_word +'s') in COMPOUND_WORDS:
			_cmp_list.pop()
			_cmp_list.pop()
			_cmp_list.append(_2_word)
		if _3_word in COMPOUND_WORDS:
			_cmp_list.pop()
			_cmp_list.pop()
			_cmp_list.pop()
			_cmp_list.append(_3_word)

def read_list(_file):
	with open(_file, 'rb') as F:
		_data = F.read()
	_list = re.split(', ', _data)
	return _list

file_name = sys.argv[1]
print "reading data"
#corpus = pd.read_csv(filename)
#corpus['content'] = corpus['content'].apply(lambda x: clean_string(x))
#corpus['title'] = corpus['title'].apply(lambda x: clean_string(x))
#corpus.to_csv("clean_test.csv", sep=',')

compound_file = "compound_word.txt"

COMPOUND_LIST = read_list(compound_file)
[ COMPOUND_WORDS.add(word) for word in COMPOUND_LIST ]

in_file =  open(file_name, 'rb')
data = csv.DictReader(in_file)

print "Processing Data"

doc_list = []
for doc in data:
	titles = token_string(doc['title'])
	contents = token_string(doc['content'])
	doc_list.append(titles)
	doc_list.append(contents)
	cmp_titles = []
	cmp_contents = []
	
	doc2cmp(titles, cmp_titles)
	doc2cmp(contents, cmp_contents)
	doc_list.append(cmp_titles)
	doc_list.append(cmp_contents)
	
del data
in_file.close()

print "Writing Data"
with open("clean_text.txt", 'wb+') as F:
	for doc in doc_list:
		for word in doc:
			F.write(word+' ')
		F.write('\n')

