import mdp.utils
import pylab
import numpy as np
import cPickle
import Oger
import matplotlib.cm as cm

epochs =1

print 'Loading data...'
data = cPickle.load(open('question.p')) #dictionary file

train_len = 1000 #len(data['trainquestions'])
test_len = 500 #len(data['testquestions'])

test_data = data['trainquestions']
test_labels = data['testlabels']
train_data = data['trainquestions']
train_labels = data['trainlabels'] 

w_train_data = train_data[:train_len]
w_train_labels = train_labels[:train_len]
w_test_data = test_data[:test_len]
w_test_labels = test_labels[:test_len]


reservoir = Oger.nodes.ReservoirNode(input_dim=64, output_dim=200)
readout = Oger.nodes.RidgeRegressionNode(input_dim = 200, output_dim=6)

Oger.utils.make_inspectable(Oger.nodes.ReservoirNode)

flow = mdp.Flow([reservoir, readout], verbose=1)
#flow = mdp.Flow([reservoir], verbose=1)

x=list()
xb=list()

'''creating list iterable for Oger'''

for j in range(0,train_len):
	y=list()
	yb = list()
	for i in range(0, len(w_train_data[j])):
		z = list()
		zb=list()

		for k in range(64):
			z.append(w_train_data[j][i][k])
		
		temp=[0 for p in range(6)]
		temp[w_train_labels[j]] = 1	

		for kb in range(6):
			zb.append(temp[kb]) #can do this appending based on when the training has to eb enforced	
			
		y.append(z)
		yb.append(zb)

	x.append(y)
	xb.append(yb)


flow.train([x[0], [(np.array(x[0]),np.array(xb[0]))]])
#flow.train([x[1], [(np.array(x[1]),np.array(xb[1]))]])


print 'end of program'
