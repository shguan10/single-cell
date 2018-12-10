
import pandas as pd
import numpy as np

K = 10

def get_to_drop(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	labels = store['labels']
	store.close()
	unique_labels = labels.unique()
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)
	fl['studies'] = fl.index.to_series()
	fl['studies'] = fl['studies'].apply(lambda s: int(s[:s.find("_")]))
	study_lists = [[], [], []]
	for label in unique_labels:
		relevant_rows = fl.loc[fl['labels'] == label]
		relevant_studies = relevant_rows['studies'].unique()
		for i in range(3):
			study_lists[i].append(np.random.choice(relevant_studies, 1)[0])
	to_drop = []
	for i in range(3):
		new_table = fl[~fl['studies'].isin(study_lists[i])]
		possible_studies = new_table['studies'].unique()
		to_drop.append(np.random.choice(possible_studies, 7))
	return to_drop
	#fl = fl[~fl.labels.str.contains("UBERON")]
	#studies = feat_mat_df.index.to_series().apply(
	#	lambda s: int(s[:s.find("_")])).unique()
	#print(fl)
	#return fl, studies

def random_split(studies):
	np.random.shuffle(studies)
	base_len = len(studies) // K * K
	rem_len = len(studies) - base_len
	base, remainder = studies[:base_len], studies[base_len:]
	studies = np.split(base, K)
	for i in range(len(remainder)):
		studies[i] = np.append(studies[i], remainder[i])
	return studies

def leave_out(fl, studies_to_leave_out):
	fl['studies'] = fl.index.to_series()
	fl['studies'] = fl['studies'].apply(lambda s: int(s[:s.find("_")]))
	fl_train = fl[~fl.studies.isin(studies_to_leave_out)]
	fl_train = fl_train.drop(labels = 'studies', axis=1)

	fl_test = fl[fl.studies.isin(studies_to_leave_out)]
	fl_test = fl_test.drop(labels = 'studies', axis=1)

	return fl_train, fl_test

'''
fl, studies = get_joint_df('../ml_10701_ps5_data/train_data.h5')
train, test = leave_out(fl, studies)
print(train)
print(test)
studies = random_split(studies)
new_train, new_test = leave_out(fl, studies[0])
print(new_train)
print(new_test)
assert(len(train) + len(test) == len(new_train) + len(new_test))
'''
studies = get_to_drop('../ml_10701_ps5_data/train_data.h5')
print(studies)
import pickle
with open('leave_k=%d_out_studies'%K, 'wb') as f:
	pickle.dump(studies, f)


