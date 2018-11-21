
### ---- Dimensionality Reduction Models ---- ###

# PCA (unsupervised)

def PCA(x_train, x_test):
	from sklearn.decomposition import PCA
	pca = PCA(n_components = 100)
	print("PCA starting")
	x_train = pca.fit_transform(x_train)
	print("PCA trained")
	x_test = pca.transform(x_test)
	print("PCA done")
	return x_train, x_test

# ICA (unsupervised)

def ICA(x_train, x_test):
	from sklearn.decomposition import FastICA
	ica = FastICA(n_components = 10)
	x_train = ica.fit_transform(x_train)
	print("ICA trained")
	x_test = ica.transform(x_test)
	print("ICA done")
	return x_train, x_test

# My dim reduction algo (supervised)

def myReducedDimMain():
	x_train_df, y_train_df = getXYasDataFrames(
		'../ml_10701_ps5_data/train_data.h5')

	x_test_df, y_test_df = getXYasDataFrames(
		'../ml_10701_ps5_data/test_data.h5')

	print("Starting reduced dim")

	x_train_df, x_test_df = myReducedDim(x_train_df, y_train_df, x_test_df)

	x_train, x_test = x_train_df.values, x_test_df.values

	return x_train, x_test

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



