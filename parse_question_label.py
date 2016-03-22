from gensim import models
import numpy as np
import pickle as cPickle

#Label index reference

ABBR = 0 #(Abbreviation)
ENTY = 1 #(Entity)
DESC = 2 #(Description)
HUM = 3 #(Human)
LOC = 4 #(Location)
NUM = 5 #(Numeric)


#reading the training file
print 'Training file read begin'
fh= open('train_1000.label') #train label file
print 'Training file read complete'
content = fh.readlines()
num_lines = len (content)
#print('Number of lines', num_lines) #debugging
#num_lines = 1 #debugging

#reading the corpus vector file
model = models.Word2Vec.load_word2vec_format('wiki_vectors.bin', binary = True) #gensim 

#creating working arrays
nullarray = np.zeros(64) 
avgarray = np.empty(64)
try:
	avgarray.fill(np.average(model['uncommon'])) # generating an "average" array
except KeyError as e: 
	print("Error! try another word for generating average...")
	

#print(avgarray) #debugging

#splitting class from question for training
train = list()
for i in range(0,num_lines):
	train.append(content[i].split())
        cls = train[i][0].split(":",1)[0] #replacing type of question with class number
        if cls == 'ABBR':  train[i][0] = ABBR      
	elif cls == 'ENTY':  train[i][0] = ENTY 
	elif cls == 'DESC':  train[i][0] = DESC 
	elif cls == 'HUM':  train[i][0] = HUM 
	elif cls == 'LOC':  train[i][0] = LOC 
	elif cls == 'NUM':  train[i][0] = NUM 

	for tra in range(1,len(train[i])): #ignoring the first class variable
		try:
			train[i][tra] = (model[train[i][tra].lower()])
		except KeyError as e: 
			if train[i][tra] == '?':
				train[i][tra] = nullarray
			else:
				train[i][tra] = avgarray

train_labels = list()

for i in range(0,num_lines):
	train_labels.append(train[i].pop(0))


print 'Testing file read begin'
fh= open('TREC_10.label') #test label file
print 'Testing file read complete'
content = fh.readlines()
num_lines = len (content)
#print('Number of lines', num_lines) #debugging
#num_lines = 1 #debugging

#reading the corpus vector file
model = models.Word2Vec.load_word2vec_format('wiki_vectors.bin', binary = True) #gensim 

#creating working arrays
nullarray = np.zeros(64) 
avgarray = np.empty(64)
try:
	avgarray.fill(np.average(model['uncommon'])) # generating an "average" array
except KeyError as e: 
	print("Error! try another word for generating average...")
	

#print(avgarray) #debugging

#splitting class from question for testing
test = list()
for i in range(0,num_lines):
	test.append(content[i].split())
        cls = test[i][0].split(":",1)[0] #replacing type of question with class number
        if cls == 'ABBR':  test[i][0] = ABBR      
	elif cls == 'ENTY':  test[i][0] = ENTY 
	elif cls == 'DESC':  test[i][0] = DESC 
	elif cls == 'HUM':  test[i][0] = HUM 
	elif cls == 'LOC':  test[i][0] = LOC 
	elif cls == 'NUM':  test[i][0] = NUM 

	for tra in range(1,len(test[i])): #ignoring the first class variable
		try:
			test[i][tra] = (model[test[i][tra].lower()])
		except KeyError as e: 
			if test[i][tra] == '?':
				test[i][tra] = nullarray
			else:
				test[i][tra] = avgarray

test_labels = list()

for i in range(0,num_lines):
	test_labels.append(test[i].pop(0))


data = dict()

data['trainquestions'] = train	
data['testquestions'] = test
data['testlabels'] = test_labels
data['trainlabels'] = train_labels


cPickle.dump(data, open('question.p', 'w'), protocol=2)


