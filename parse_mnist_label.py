from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pickle as cPickle

white = 0.001

mndata = MNIST('mnist/') #directiry holding the MNIST dataset

mndata.load_training()
mndata.load_testing()

train_data = np.reshape(mndata.train_images, (60000, 28, 28))
test_data = np.reshape(mndata.test_images, (10000, 28, 28))

train_data = train_data / 255.
test_data = test_data / 255.

for x in range(0,len(train_data)):
	train_data[x] = np.rot90(train_data[x], 3)

for x in range(0,len(test_data)):
	test_data[x] = np.rot90(test_data[x], 3)

trn_labels = list()
tst_labels = list()

white_space=[0 for p in range(11)]
white_space[10] = 1

t = 2

for j in range(0, len(train_data)):
#	print 'img-', j
	for i in range(0, 28):
		ind1 = 0
		if np.argmax(train_data[j][i]>white)>0:
			ind1 = i

			break
	for i in reversed(range(0,28)):
		ind2 = 0
		if np.argmax(train_data[j][i]>white)>0:
			ind2 = i

			break 
	temp = list()
	zeros=[0 for p in range(11)]

	for t_num in range(0,28):
		if t_num<ind1 or t_num>ind2:
			temp.append(white_space)
		else:
			zeros[mndata.train_labels[j]] = 1
			temp.append(zeros)
	trn_labels.append(temp)

for j in range(0, len(test_data)):
#	print 'img-', j
	for i in range(0, 28):
		ind1 = 0
		if np.argmax(test_data[j][i]>white)>0:
			ind1 = i

			break
	for i in reversed(range(0,28)):
		ind2 = 0
		if np.argmax(test_data[j][i]>white)>0:
			ind2 = i

			break 
	temp = list()
	zeros=[0 for p in range(11)]

	for t_num in range(0,28):
		if t_num<ind1 or t_num>ind2:
			temp.append(white_space)
		else:
			zeros[mndata.test_labels[j]] = 1
			temp.append(zeros)
	tst_labels.append(temp)

'''
print(np.asarray(train_data).shape) #debugging
print(np.asarray(train_labels).shape) #debugging
print(np.asarray(test_data).shape) #debugging
print(np.asarray(test_labels).shape) #debugging


t =0 #debugging

for i in range(0,t):
	print(train_labels[i])
	plt.imshow(train_data[i])
	plt.gray()
	plt.show()
'''

data = dict()

data['traindatamnist'] = train_data
data['trainlabelsmnist'] = trn_labels
data['testdatamnist'] = test_data
data['testlabelsmnist'] = tst_labels
data['testlabelsnumbermnist'] = mndata.test_labels 
data['trainlabelsnumbermnist'] = mndata.train_labels

cPickle.dump(data, open('mnistall.p', 'w'), protocol=2)
