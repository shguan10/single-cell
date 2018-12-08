import pandas as pd
import numpy as np

### ---- Preprocessing ---- ###

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
		#one_hot = [0] * len(L)
		#one_hot[i] = 1
		d[L[i]] = i #np.array(one_hot)
	return d

def getXandYandDict(filename):
	# get x and y
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	x = feat_mat_df.values
	print("shapes", x.shape)
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

def main():
	x_train, y_train, d = getXandYandDict('../ml_10701_ps5_data/train_data.h5')
	y_train = make_np_array(y_train)

	print(x_train, y_train, d)

	x_test, y_test = getXandYwithDict('../ml_10701_ps5_data/test_data.h5', d)
	y_test = make_np_array(y_test)

	print(x_test, y_test)

	return x_train, y_train, x_test, y_test

if __name__ == '__main__':
	main()
