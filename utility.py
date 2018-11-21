### ---- Utility Functions ---- ###

def confusionMatrix(y_pred, y_test):
	matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
	np.set_printoptions(threshold = np.nan)
	print(matrix)
	np.set_printoptions()

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