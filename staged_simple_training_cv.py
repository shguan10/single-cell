'''
all cell_types: 
'CL:0000353 blastoderm cell' 'UBERON:0002107 liver'
 'CL:0000057 fibroblast' 'CL:0002322 embryonic stem cell'
 'CL:0000081 blood cell' 'UBERON:0000115 lung epithelium'
 'CL:0001056 dendritic cell, human' 'CL:0000746 cardiac muscle cell'
 'UBERON:0001851 cortex' 'CL:0002034 long term hematopoietic stem cell'
 'CL:0002033 short term hematopoietic stem cell'
 'CL:0000037 hematopoietic stem cell' 'CL:1000497 kidney cell'
 'CL:0008019 mesenchymal cell' 'UBERON:0000044 dorsal root ganglion'
 'CL:0002365 medullary thymic epithelial cell' 'UBERON:0000473 testis'
 'UBERON:0000992 female gonad' 'UBERON:0000922 embryo'
 'UBERON:0002048 lung' 'CL:0000137 osteocyte'
 'UBERON:0001898 hypothalamus' 'UBERON:0001997 olfactory epithelium'
 'CL:0002321 embryonic cell' 'CL:0002319 neural cell'
 'UBERON:0004129 growth plate cartilage' 'UBERON:0001891 midbrain'
 'UBERON:0002038 substantia nigra' 'UBERON:0000007 pituitary gland'
 'CL:0000763 myeloid cell' 'CL:0000540 neuron' 'UBERON:0000045 ganglion'
 "UBERON:0001954 Ammon's horn" 'CL:0000127 astrocyte'
 'CL:0000163 endocrine cell' 'UBERON:0000955 brain'
 'UBERON:0000966 retina' 'UBERON:0002435 striatum'
 'UBERON:0010743 meningeal cluster' 'CL:0000169 type B pancreatic cell'
 'UBERON:0001264 pancreas' 'CL:0000084 T cell'
 'UBERON:0001003 skin epidermis'
 'UBERON:0001902 epithelium of small intestine' 'CL:0000235 macrophage'
 'CL:0000192 smooth muscle cell'
 '''

from combined_hierarchy import *

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import pickle

def get_cell_superclass(s):
	if s in superclass_dict:
		index = superclass_dict[s]
		arr = [0] * len(cell_superclasses)
		arr[index] = 1
		return np.array(arr)
	else:
		print(1/0)
		return bad_phrase

def make_np_array(y):
	L = []
	for i in y:
		L.append(i)
	L = np.array(L)
	return L

def getXandYandDict(filename):
	x, y_series = getXandYandDictNoSuperclass(filename)
	y = y_series.apply(lambda s: get_cell_superclass(s))#.values?
	print(y.value_counts())

	print("Got x and y")

	return x, y.values

def getXandYandDictNoSuperclass(filename):
	# get x and y
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	#x = feat_mat_df.values
	labels = store['labels']
	store.close()
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)

	print(fl['labels'])
	fl = fl[~fl.labels.str.contains(bad_phrase)]
	print(fl)

	x_df = fl.drop(labels = 'labels', axis=1)
	print(x_df)
	x = x_df.values

	y_series = fl['labels']
	print(y_series)

	return x, y_series

def pprint_errors(y_pred, y_test):
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

# this function modified from leave_k_out
def get_joint_df(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	labels = store['labels']
	store.close()
	labels = labels.apply(lambda s: get_cell_superclass(s))
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

def main_stage_1():
	with open('leave_k=10_out_studies', 'rb') as f:
		studies = pickle.load(f)

	results = []
	fl_overall, d = get_joint_df('../ml_10701_ps5_data/train_data.h5')
	for i in range(len(studies)):
		fl_train, fl_test = leave_out(fl_overall, studies[i])
		x_tra, y_tra = split_XY(fl_train)
		x_val, y_val = split_XY(fl_test)
		print(x_tra.shape, y_tra.shape, x_val.shape, y_val.shape)
		new_loss = validate(x_tra, y_tra, x_val, y_val)
		results.append(new_loss)

	print(results)

	with open('stage_1_k=10_out_results', 'wb') as f:
		pickle.dump(results, f)


def validate(x_train, y_train, x_test, y_test):
	# COPY PASTED FROM SIMPLE_TRAINING.PY

	noOfTrainingSamples, noOfFeatures = x_train.shape
	assert(noOfFeatures == x_test.shape[1])
	noOfClasses = y_train.shape[1]
	assert(noOfClasses == y_test.shape[1])

	'''
	from dim_red_models import *
	x_train, x_test = myReducedDimMain()
	'''

	sc = StandardScaler()
	x_train = sc.fit_transform(X = x_train)
	x_test = sc.transform(X = x_test)

	print(x_train.shape, x_test.shape)

	print(x_train, x_test)
	print(type(x_train))
	print(type(x_test))
	#ssert(0)
	print("Normalized features")

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

	matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
	np.set_printoptions(threshold = np.nan)
	print(matrix)

	pprint_errors(y_pred, y_test)

	with open(stage_1_filename, 'wb') as f:
		pickle.dump((classifier, sc), f)

	return loss
'''
d_new = {}
for i in range(len(d)):
	d_new[i] = np.argmax(d[i])
print(d_new)
'''

if __name__ == '__main__':
	main_stage_1()

