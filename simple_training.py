import pandas as pd
import numpy as np

def normalize_rows(x):
	return x / (x.sum(axis=1)[:, np.newaxis])

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

def getXandYandDict(filename):
	# get x and y
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	x = feat_mat_df.values
	labels = store['labels']
	d = getDict(labels)
	y = labels.apply(lambda s: d[s]).values
	print("Critical info here")
	print(labels.apply(lambda s: d[s]).apply(
		lambda s: np.argmax(s)).value_counts())
	assert(x.shape[0] == y.shape[0])
	return x, y, d

def getXandYwithDict(filename, d):
	# get x and y
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	x = feat_mat_df.values
	labels = store['labels']
	y = labels.apply(lambda s: d[s]).values
	assert(x.shape[0] == y.shape[0])
	return x, y

x_train, y_train, d = getXandYandDict(#
	'../ml_10701_ps5_data/train_data.h5')#'../ml_10701_ps5_data/train_data.h5')
print("Normalize training by row")
#x_train = normalize_rows(x_train)
print(x_train.sum(axis=1))
y_train = make_np_array(y_train)

print(x_train, y_train, d)

x_test, y_test = getXandYwithDict('../ml_10701_ps5_data/test_data.h5', d)
print("Normalize test by row")
#x_test = normalize_rows(x_test)
print(x_test.sum(axis=1))
y_test = make_np_array(y_test)

print(x_test, y_test)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y,
#	test_size = 0.2)

noOfTrainingSamples, noOfFeatures = x_train.shape
assert(noOfFeatures == x_test.shape[1])
noOfClasses = y_train.shape[1]
assert(noOfClasses == y_test.shape[1])
#uncomment here
#print("Split data")
# pre-processing and NN code based on example in
# https://medium.com/@pushkarmandot/
# build-your-first-deep-learning-neural-network-model-using-keras-in-
# python-a90b5864116d

# preprocess x data

#print(x_train.shape, x_test.shape)

from dim_red_models import *
x_train, x_test = PCA(x_train, x_test)


#from dim_red_models import *
#x_train, x_test = myReducedDimMain()

# x_train, x_test, y_train = getRelevantFeatures(x_train, y_train, x_test)

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

x = classifier.fit(x_train, y_train, batch_size = 10, epochs = 5,
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
false_pos_L = []
class_no_L = []

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
	false_pos_L.append(false_positives)
	class_no_L.append(i)

for x in matrix:
	print(x)

print(false_pos_L)
print(class_no_L)
'''
d_new = {}
for i in range(len(d)):
	d_new[i] = np.argmax(d[i])
print(d_new)
'''