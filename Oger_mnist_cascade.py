#Code for two cascading reservoirs

import mdp.utils
import numpy as np
import cPickle
import Oger
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from PIL import Image
import random
from random import randrange
import os


train_len = 60000 # should be multiple of batch
test_len = 10000

batch = 500

resnode = 500
numconn = 5

if __name__ == "__main__":
	print 'Loading data...'
	data = cPickle.load(open('mnistall.p','r')) #dictionary file

	wt_in = [[0 for x in range(28)] for x in range(resnode)]

	for i in range(resnode):
		x = random.sample(range(28),numconn)
		for t in x:
			wt_in[i][t]=random.uniform(-1,1)

        wt_cas = [[0 for x in range(11)] for x in range(resnode)]

        for i in range(resnode):
                x = random.sample(range(11),numconn)
                for t in x:
                        wt_in[i][t]=random.uniform(-1,1)



	wt_rec = [[0 for x in range(resnode)] for x in range(resnode)]

	for i in range(resnode):
		x = random.sample(range(resnode),numconn)
		for t in x:
		        wt_rec[t][i]=random.uniform(-1,1)

	trn_input=data['traindatamnist'] 
	train_input = trn_input[:train_len]

	trn_target=data['trainlabelsmnist'] 
	train_target = trn_target[:train_len]
	
	tst_input=data['testdatamnist']
	test_input = tst_input[:test_len]

	tst_target=data['testlabelsmnist']
	test_target = tst_target[:test_len]

	tst_label=data['testlabelsnumbermnist']
	test_label = tst_label[:test_len]
	
	reservoir1 = Oger.nodes.LeakyReservoirNode(input_dim=28, output_dim=resnode, w = np.array(wt_rec), w_in= np.array(wt_in), leak_rate=0.22, input_scaling=0.28, spectral_radius = 0.65, dtype='f') # First reservoir
	readout1 = Oger.nodes.RidgeRegressionNode(input_dim = resnode, output_dim=11, dtype='f') # there are 10 digits plus one class for the white space before and after each digit
	reservoir2 = Oger.nodes.LeakyReservoirNode(input_dim=11, output_dim=resnode, w = np.array(wt_rec), w_in= np.array(wt_cas), leak_rate=0.22, input_scaling=0.28, spectral_radius = 0.65, dtype='f') #Second reservoir
        readout2 = Oger.nodes.RidgeRegressionNode(input_dim = resnode, output_dim=11, dtype='f') # there are 10 digits plus one class for the white space before and after each digit

	print 'training begins'

	srt = 0 #for managing indexing for every iteration
	for ep in range(train_len/batch):
		print '-------- Batch ' + str(ep) +' with range '+ str(srt) +'-' + str(srt+batch) + '------------------'
		flow = mdp.CheckpointFlow([reservoir1, readout1, reservoir2, readout2], verbose=1)
		readout1.verbose = 1
		readout2.verbose = 1

		#Configuration here is unclear
		flow.train([np.array(train_input[srt:srt+batch]), zip(np.array(train_input[srt:srt+batch]), np.array(train_target[srt:srt+batch])), np.array(train_input[srt:srt+batch]), zip(np.array(train_input[srt:srt+batch]),np.array(train_target[srt:srt+batch]))], [None,mdp.CheckpointSaveFunction('readout1.pic',stop_training = 0,protocol=2), None, mdp.CheckpointSaveFunction('readout2.pic',stop_training = 0,protocol=2)])

		srt = srt+batch
		fl1 = file('readout1.pic')
		fl2 = file('readout2.pic')
		readout1 = cPickle.load(fl1)
		readout2 = cPickle.load(fl2)
		fl2.close()
		fl1.close()
	
	print 'training ends'

	Error_rate=0
	for ts_inp,ts_lb in zip(test_input,test_label):
		Y=flow(ts_inp)
		recognized=np.argmax(np.sum(Y[:,:-1],axis=0)) # Winner-Takes-All over digits only (class 11 which is the white space excluded from the calculation)
		Error_rate+=(recognized!=ts_lb) # +1 if the recognitized digit differs from the label
	print("Error rate is %.2f%%" % (Error_rate*100.0/len(test_input)))

print 'end of program'

