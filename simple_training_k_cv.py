import pandas as pd
import numpy as np

# only 3 epochs

def make_np_array(y):
	L = []
	for i in y:
		L.append(i)
	L = np.array(L)
	return L

def getDict(labels):
	L = labels.unique()
	d = {}
	for i in range(len(L)):
		one_hot = [0] * len(L)
		one_hot[i] = 1
		d[L[i]] = np.array(one_hot)
	return d

# this function modified from leave_k_out
def get_joint_df(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	labels = store['labels']
	d = getDict(labels)
	labels = labels.apply(lambda s: d[s])
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)
	return fl, d

# this function from leave_k_out
def leave_out(fl, studies_to_leave_out):
	fl['studies'] = fl.index.to_series()
	fl['studies'] = fl['studies'].apply(lambda s: int(s[:s.find("_")]))
	fl_train = fl[~fl.studies.isin(studies_to_leave_out)]
	fl_train = fl_train.drop(labels = 'studies', axis=1)

	fl_test = fl[fl.studies.isin(studies_to_leave_out)]
	fl_test = fl_test.drop(labels = 'studies', axis=1)

	return fl_train, fl_test

def split_XY(fl):
	x = fl.drop(labels = 'labels', axis=1).values
	y = make_np_array(fl['labels'].values)
	return x, y

import pickle

def main():
	with open('leave_k=10_out_studies', 'rb') as f:
		studies = pickle.load(f)

	results = []
	fl_overall, d = get_joint_df('../ml_10701_ps5_data/train_data.h5')
	for i in range(3):#len(studies)):
		fl_train, fl_test = leave_out(fl_overall, studies[i])
		x_tra, y_tra = split_XY(fl_train)
		x_val, y_val = split_XY(fl_test)
		print(x_tra.shape, y_tra.shape, x_val.shape, y_val.shape)
		new_loss = test_model(x_tra, y_tra, x_val, y_val)
		results.append(new_loss)
		print("Done with i = %dth iteration"%i)

	print(results)

	with open('vanilla_k=10_out_results', 'wb') as f:
		pickle.dump(results, f)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y,
#	test_size = 0.2)

# pre-processing and NN code based on example in
# https://medium.com/@pushkarmandot/
# build-your-first-deep-learning-neural-network-model-using-keras-in-
# python-a90b5864116d

'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 796)
print("PCA starting")
x_train = pca.fit_transform(x_train)
print("PCA trained")
x_test = pca.transform(x_test)
print("PCA done")
'''

# x_train, x_test, y_train = getRelevantFeatures(x_train, y_train, x_test)

def test_model(x_train, y_train, x_test, y_test):
	noOfTrainingSamples, noOfFeatures = x_train.shape
	assert(noOfFeatures == x_test.shape[1])
	noOfClasses = y_train.shape[1]
	assert(noOfClasses == y_test.shape[1])

	'''
	from dim_red_models import *
	x_train, x_test = myReducedDimMain()
	'''

	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	x_train = sc.fit_transform(X = x_train)
	x_test = sc.transform(X = x_test)

	print(x_train.shape, x_test.shape)

	print(x_train, x_test)
	print(type(x_train))
	print(type(x_test))
	#ssert(0)
	print("Normalized features")

	import keras
	from keras.models import Sequential
	from keras.layers import Dense
	from keras import optimizers

	hiddenNodes = 100

	classifier = Sequential()

	classifier.add(Dense(output_dim = hiddenNodes, input_dim = noOfFeatures,
		activation = 'tanh', init = 'glorot_uniform'))
	#classifier.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
	#	activation = 'tanh'))
	classifier.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
		activation = 'softmax'))

	# sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
	classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
		metrics = ['accuracy'])

	x = classifier.fit(x_train, y_train, batch_size = 10, epochs = 3,
		verbose = 1 )
	#	validation_data = (x_test, y_test))

	"Fit training samples to classifier"

	loss = classifier.evaluate(x_test, y_test,
		verbose = 1)

	"Evaluated test samples"

	print(loss)
	print(classifier.metrics_names)

	y_pred = classifier.predict(x_test)

	from sklearn import metrics
	matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
	np.set_printoptions(threshold = np.nan)
	print(matrix)

	matrix = []
	y_pred = np.argmax(y_pred, axis = 1)
	y_test = np.argmax(y_test, axis = 1)
	for i in set(y_pred):
		false_positives, false_negatives = 0, 0
		true_positives = 0
		for j in range(len(y_test)):
			if (y_pred[j] == i) and (y_test[j] == i):
				true_positives += 1
			elif (y_pred[j] == i) and not (y_test[j] == i):
				false_positives += 1
			elif (not (y_pred[j] == i)) and (y_test[j] == i):
				false_negatives += 1
		total_no = true_positives + false_negatives
		if total_no != 0:
			matrix.append((i, total_no, true_positives / total_no,
				false_positives / total_no, false_negatives / total_no))

	for x in matrix:
		print(x)
	return loss

main()
'''
d_new = {}
for i in range(len(d)):
	d_new[i] = np.argmax(d[i])
print(d_new)
'''