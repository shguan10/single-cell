# STAGE 2

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import pickle

from staged_simple_training import *

def get_keys(i):
	keys = []
	for x in superclass_dict:
		if superclass_dict[x] == i:
			keys.append(x)
	return keys

def init_dfs(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	labels = store['labels']
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)

	dfs_x = []
	dfs_y = []
	all_keys = []

	for i in range(len(cell_superclasses)):
		
		keys = get_keys(i)

		def make_one_hot(i):
			res = np.array([0] * len(keys))
			res[i] = 1
			return res

		next_df_xy = fl[fl.labels.isin(keys)]
		next_df_x = next_df_xy.drop(labels = 'labels', axis=1)
		next_df_y = next_df_xy['labels'].apply(
			lambda s: keys.index(s)).apply(make_one_hot)
		dfs_x.append(next_df_x)
		dfs_y.append(next_df_y)
		all_keys.append(keys)
	
	return dfs_x, dfs_y, all_keys

def train_one_nn_scaler(df_x, df_y, keys):
	x_train, y_train = df_x.values, df_y.values
	y_train = make_np_array(y_train)
	noOfTrainingSamples, noOfFeatures = x_train.shape
	noOfClasses = y_train.shape[1]

	# scale first
	scaler = StandardScaler()
	x_train = scaler.fit_transform(X = x_train)

	# create network
	hidden_nodes = 15
	nn = Sequential()
	nn.add(Dense(output_dim = hidden_nodes, input_dim = noOfFeatures,
		activation = 'tanh', init = 'glorot_uniform'))
	nn.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
		activation = 'softmax'))

	# train network
	# sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
	nn.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
		metrics = ['accuracy'])

	x = nn.fit(x_train, y_train, batch_size = 10, epochs = 5,
		verbose = 1 )
	loss = nn.evaluate(x_train, y_train,
		verbose = 1)
	print("Training losses", loss)
	print(nn.metrics_names)
	return nn, scaler


def train_nns_scalers(dfs_x, dfs_y, all_keys):
	nns = []
	scalers = []
	for i in range(len(all_keys)):
		nn, scaler = train_one_nn_scaler(dfs_x[i], dfs_y[i], all_keys[i])
		print("Trained %dth classifier"%i)
		nns.append(nn)
		scalers.append(scaler)
	return nns, scalers

def main_stage_2():
	print("About to initialize dataframes")
	dfs_x, dfs_y, all_keys = init_dfs(
		'../oversampled_train_data.h5')#'../ml_10701_ps5_data/train_data.h5')
	print("About to train nns")
	nns, scalers = train_nns_scalers(dfs_x, dfs_y, all_keys)

	with open(stage_2_filename, 'wb') as f:
		pickle.dump((nns, scalers, all_keys), f)

	return nns, scalers, all_keys

if __name__ == '__main__':
	main_stage_2()

