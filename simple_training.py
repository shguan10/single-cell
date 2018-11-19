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

x_train, y_train, d = getXandYandDict('../ml_10701_ps5_data/train_data.h5')
print("Normalize training by row")
x_train = normalize_rows(x_train)
print(x_train.sum(axis=1))
y_train = make_np_array(y_train)

print(x_train, y_train, d)

x_test, y_test = getXandYwithDict('../ml_10701_ps5_data/test_data.h5', d)
print("Normalize test by row")
x_test = normalize_rows(x_test)
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

print("Split data")
# pre-processing and NN code based on example in
# https://medium.com/@pushkarmandot/
# build-your-first-deep-learning-neural-network-model-using-keras-in-
# python-a90b5864116d

# preprocess x data

print(x_train.shape, x_test.shape)
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
'''
import keras
from keras.models import Sequential
from keras.layers import Dense

hiddenNodes = 796

classifier = Sequential()

classifier.add(Dense(output_dim = hiddenNodes, input_dim = noOfFeatures,
	activation = 'tanh', init = 'glorot_uniform'))
classifier.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
	activation = 'softmax'))

classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
	metrics = ['accuracy'])

x = classifier.fit(x_train, y_train, batch_size = 10, epochs = 2,
	verbose = 1 )
#	validation_data = (x_test, y_test))

"Fit training samples to classifier"

loss = classifier.evaluate(x_test, y_test,
	verbose = 1)

"Evaluated test samples"

print(loss)
print(classifier.metrics_names)




