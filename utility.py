import numpy as np
import pandas as pd

### ---- Utility Functions ---- ###

def confusionMatrix(y_pred, y_test):
	matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
	np.set_printoptions(threshold = np.nan)
	print(matrix)
	np.set_printoptions()

def up_sample(x_df, y_series):
	fl = pd.concat([x_df, pd.DataFrame(y_series.rename('labels'))],
		axis = 1)
	y_values = y_series.value_counts()
	y_max = y_values.max()
	labels = y_series.unique()

	print(y_values)
	print(y_max)

	new_df_x, new_y_series = None, None

	for label in labels:
		print(label)
		# I tried with 500 and 1000, changed to 1500 now, need to run it
		# on a good computer
		amount_needed = 500
		correct_label_df = fl[fl.labels == label]
		to_replace = y_values[label] < amount_needed
		extra = correct_label_df.sample(amount_needed, replace = to_replace)
		if new_df_x is None:
			new_df_x = extra.drop(labels = 'labels', axis = 1)
		else:
			new_df_x = pd.concat([new_df_x,
				extra.drop(labels = 'labels', axis=1)], axis = 0)
		if new_y_series is None:
			new_y_series = extra['labels']
		else:
			new_y_series = new_y_series.append(extra['labels'])

	return new_df_x, new_y_series

store = pd.HDFStore('../ml_10701_ps5_data/train_data.h5')
feat_mat_df = store['rpkm']; labels = store['labels']
store.close()
new_x, new_y = up_sample(feat_mat_df, labels)
print(new_y.value_counts())
print(new_x.values.shape)
print(new_y.values.shape)

new_x.to_hdf('../oversampled_train_data_500.h5', key = 'rpkm', mode = 'w')
new_y.to_hdf('../oversampled_train_data_500.h5', key = 'labels')


'''
def nicelyPrintedErrors(y_pred, y_test):
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
	print("type no, total, true pos ratio, false pos ratio, false neg ratio")
	for x in matrix:
		print(x)
'''