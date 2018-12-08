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

x_train, y_train, d = getXandYandDict('../ml_10701_ps5_data/train_data.h5')

y_train = make_np_array(y_train)

print(x_train, y_train, d)

x_test, y_test = getXandYwithDict('../ml_10701_ps5_data/test_data.h5', d)

y_test = make_np_array(y_test)

print(x_test, y_test)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y,
#	test_size = 0.2)

noOfTrainingSamples, noOfFeatures = x_train.shape
assert(noOfFeatures == x_test.shape[1])
#noOfClasses = y_train.shape[1]
#assert(noOfClasses == y_test.shape[1])

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

### ---- Models ---- ###


# PCA

'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
print("PCA starting")
x_train = pca.fit_transform(x_train)
print("PCA trained")
x_test = pca.transform(x_test)
print("PCA done")
'''

# ICA

'''
from sklearn.decomposition import FastICA
ica = FastICA(n_components = 10)
x_train = ica.fit_transform(x_train)
print("ICA trained")
x_test = ica.transform(x_test)
print("ICA done")
'''

# SVM

'''
from sklearn import svm
lin_clf = svm.SVC(kernel = 'rbf')
lin_clf.fit(x_train, y_train)
print("Fit")
print(lin_clf.score(x_test, y_test))
'''

# Random Forest

'''
from sklearn.ensemble import RandomForestClassifier
print("Starting")
clf = RandomForestClassifier(n_estimators = 50, max_depth = 10,
	class_weight = 'balanced')
clf.fit(x_train, y_train)
print("Fit")
print("training accuracy", clf.score(x_train, y_train))
print("test accuracy", clf.score(x_test, y_test))
'''

# AdaBoost with Decision Trees

'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6),
	n_estimators = 10)
abc.fit(x_train, y_train)
print("Trained")
print("Training accuracy", abc.score(x_train, y_train))
print("Test accuracy", abc.score(x_test, y_test))
'''
'''
# My dim reduction algo

# Inputs x_train, y_train and x_test must be dataframes and not
# np arrays
def myReducedDim(x_train, y_train, x_test):
	# returns x_train and x_test in a reduced dim

	# 0. combine x and y
	print("Step 0")
	y_train_named = y_train.rename("cell_type")
	combined = x_train.join(y_train_named)

	# 1. get mean of each gene's activity
	print("Step 1")
	means = combined.mean(axis = 0, numeric_only = True)

	# 2. for each cell type, identify the top 5 genes in terms of
	# deviation from mean - add them to the set to consider
	print("Step 2")
	to_keep = set()
	features_per_type = 20
	for cell_type in y_train.unique():
		relevant_rows = combined.loc[combined['cell_type'] == cell_type]
		relevant_means = relevant_rows.mean(axis = 0, numeric_only = True)
		dev_from_mean = abs(relevant_means - means) / means
		best_ones = dev_from_mean.nlargest(features_per_type)
		to_keep |= set(best_ones.index.values.tolist())

	# 3. discard the rest of the genes
	print("Step 3")
	to_keep = list(to_keep)
	x_train  = x_train[to_keep]
	x_test = x_test[to_keep]
	print(x_train)
	print(x_test)
	assert(x_train.columns.values.tolist() == x_test.columns.values.tolist())
	return x_train, x_test

def getXYasDataFrames(filename):
	store = pd.HDFStore(filename)
	return store['rpkm'], store['labels']

x_train_df, y_train_df = getXYasDataFrames(
	'../ml_10701_ps5_data/train_data.h5')

x_test_df, y_test_df = getXYasDataFrames(
	'../ml_10701_ps5_data/test_data.h5')

print("Starting reduced dim")
x_train_df, x_test_df = myReducedDim(x_train_df, y_train_df, x_test_df)

x_train, x_test = x_train_df.values, x_test_df.values
'''
from sklearn.ensemble import RandomForestClassifier
print("Starting")
clf = RandomForestClassifier(n_estimators = 200, max_depth = 10,
	class_weight = 'balanced')
clf.fit(x_train, y_train)
print("Fit")
print("training accuracy", clf.score(x_train, y_train))
print("test accuracy", clf.score(x_test, y_test))


