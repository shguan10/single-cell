from staged_simple_training import *
from stage_2_training import *

def main():
	with open('stage_1_models.pickle', 'rb') as f:
		stage_1_classifier, stage_1_sc = pickle.load(f)
	#main_stage_1()
	with open('stage_2_models.pickle', 'rb') as f:
		stage_2_classifiers, stage_2_scs, all_keys = pickle.load(f)
	#main_stage_2()

	x_test, y_test_series = getXandYandDictNoSuperclass(
		'../ml_10701_ps5_data/test_data.h5')
	y_test = y_test_series.values

	rows, successes = 0, 0

	print("Testing...")
	for index in range(len(x_test)):
		row = x_test[index]
		x = np.array([row])
		x_scaled_1 = stage_1_sc.transform(X = x)
		y_pred_1 = np.argmax(stage_1_classifier.predict(x_scaled_1),
			axis = 1)[0]
		classifier_2, sc_2 = (stage_2_classifiers[y_pred_1],
			stage_2_scs[y_pred_1])
		x_scaled_2 = sc_2.transform(X = x)
		y_pred_2 = np.argmax(classifier_2.predict(x_scaled_2), axis = 1)[0]
		if (len(all_keys[y_pred_1]) > y_pred_2
			and all_keys[y_pred_1][y_pred_2] == y_test[index]):
			successes += 1
		rows +=1

	print("Accuracy is %0.4f"%(successes/rows))

main()