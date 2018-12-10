
import pandas as pd
import numpy as np

K = 10

def get_joint_df(filename):
	store = pd.HDFStore(filename)
	feat_mat_df = store['rpkm']
	#print(studies)
	labels = store['labels']
	store.close()
	fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
		axis = 1)
	#fl = fl[~fl.labels.str.contains("UBERON")]
	studies = feat_mat_df.index.to_series().apply(
		lambda s: int(s[:s.find("_")])).unique()
	#print(fl)
	return fl, studies

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


fl, studies = get_joint_df('../ml_10701_ps5_data/train_data.h5')
train, test = leave_out(fl, studies)
print(train)
print(test)
studies = random_split(studies)
new_train, new_test = leave_out(fl, studies[0])
print(new_train)
print(new_test)
assert(len(train) + len(test) == len(new_train) + len(new_test))

import pickle
with open('leave_k=%d_out_CL_only_studies'%K, 'wb') as f:
	pickle.dump(studies, f)


