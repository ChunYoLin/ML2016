import sys
import re
import operator
import math
import csv

stop_list = []
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
	string = re.sub('https:[\/\.\w\-%]+', ' ', string)
	string = re.sub('&[\w]*', ' ', string)
	string = re.sub('\\\[\w]*', ' ', string)
	string = re.sub('{,*?}', ' ', string)
	str_list = re.sub('[^A-Za-z0-9\' ]',' ' ,string).split(' ')
	return [string.lower() for string in str_list \
				if bool(string) is True]

def store_dic(_word, _dict):
	_word = re.sub('\'', '', _word)
	if _word not in _dict:
		_dict[_word] = 1
	else :
		_dict[_word] += 1

def doc2bow(_list, _three_dict, _two_dict, _one_dict):
	flag=0
	last_word = ''
	last_two_word = ''
	for word in _list:
		three_word = last_two_word + '-' + word
		two_word = last_word + '-' + word
		last_word = word
		last_two_word = two_word
		if is_number(word) is True or word in stop_list or len(word) <= 2:
			flag = 0
			continue
		store_dic(word, _one_dict)	
		if flag == 0:
			flag = 1
			continue
		store_dic(two_word, _two_dict)
		if flag == 1:
			flag = 2
			continue
		store_dic(three_word, _three_dict)

file_name = sys.argv[1]
stop_file = 'stopword.txt'
file_path = re.split('\.', file_name)
file_path = re.split('/',file_path[0])
print "Reading Data"
with open(stop_file,'rb')as F:
	stop_word = F.read()
stop_list = re.split(', ', stop_word)


in_file =  open(file_name, 'rb')
data = csv.DictReader(in_file)

print "Processing Data"

word_dic = {}
two_word_dic = {}
three_word_dic = {}
for doc in data:
	titles = token_string(doc['title'])

	contents = token_string(doc['content'])

	doc2bow(titles, three_word_dic, two_word_dic, word_dic)
	doc2bow(contents, three_word_dic, two_word_dic, word_dic)

del data
in_file.close()

print "dic len : ",len(two_word_dic)
word_value = {}
for couple, value in two_word_dic.iteritems():
	first = couple.split('-')[0]
	second = couple.split('-')[1]
	prob = float(value) / float( min(word_dic[first], word_dic[second]))
	word_value[couple] = prob * value

compound_list = []
three_word_value = {}
for couple, value in three_word_dic.iteritems():
	first = couple.split('-')[0]
	second = couple.split('-')[1]
	third = couple.split('-')[2]
	prob = float(value) / float( min(word_dic[first], word_dic[second], word_dic[third]))
	three_word_value[couple] = prob * value
#'''
print "Writing Compound"

sort_3_word = sorted(three_word_value.items(), key=operator.itemgetter(1),\
					reverse=True)
comp_3_list = []
for j in range(10):
	word = sort_3_word[j][0]
	spl_word = word.split('-')
	fir = spl_word[0] + '-' + spl_word[1]
	sec = spl_word[1] + '-' + spl_word[2]
	compound_list.append(word)
	comp_3_list.append(fir)
	comp_3_list.append(sec)


sort_word = sorted(word_value.items(), key=operator.itemgetter(1),\
				reverse=True)
for j in range(160):
	word = sort_word[j][0]
	if word in comp_3_list:
		continue
			
	if (word+'s') in two_word_dic \
		and two_word_dic[word+'s'] > 250:	
		word = word + 's'
	compound_list.append(word)
#'''
with open('compound_word.txt', 'wb+') as F:
	F.write(", ".join(compound_list))

