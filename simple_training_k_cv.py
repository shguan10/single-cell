import pandas as pd
import numpy as np

# only 3 epochs
# only CL

def make_np_array(y):
	return np.array(list(y))

def getDict(labels):
	L = labels.unique()
	d = {}
	for i in range(len(L)):
		one_hot = np.zeros(len(L))
		one_hot[i] = 1
		d[L[i]] = one_hot
	return d

# this function modified from leave_k_out
def get_joint_df(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	labels = store['labels']
	store.close()
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)

	d = getDict(fl['labels'])
	fl['labels'] = fl['labels'].apply(lambda s: d[s])

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
		print("Before leaving out")
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

def test_model(x_train, y_train, x_test, y_test):
	print("Entered test model")
	noOfTrainingSamples, noOfFeatures = x_train.shape
	noOfClasses = y_train.shape[1]

	print("Before scaler")
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	x_train = sc.fit_transform(X = x_train)
	x_test = sc.transform(X = x_test)
	print("After scaler")

	print("Normalized features")

	import keras
	from keras.models import Sequential
	from keras.layers import Dense
	from keras import optimizers

	hiddenNodes = 100

	classifier = Sequential()

	classifier.add(Dense(output_dim = hiddenNodes, input_dim = noOfFeatures,
		activation = 'tanh', init = 'glorot_uniform'))
	classifier.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
		activation = 'softmax'))

	classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
		metrics = ['accuracy'])

	hist = classifier.fit(x_train, y_train, batch_size = 10, epochs = 3,
		verbose = 1, validation_data = (x_test, y_test))

	return hist.history["val_acc"][-1]

if __name__ == '__main__':
	main()