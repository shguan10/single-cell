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

def do_upsample():
  store = pd.HDFStore('../data_rpkm/train_data.h5')
  feat_mat_df = store['rpkm']
  labels = store['labels']
  new_x, new_y = up_sample(feat_mat_df, labels)
  print(new_y.value_counts())
  print(new_x.values.shape)
  print(new_y.values.shape)

  x_train = new_x
  y_train = new_y
  y=y_train
  mapping=pd.get_dummies(y_train)
  labeldict={}
  for lab,vec in zip(y,mapping.values): labeldict[lab] = vec
  y_train=mapping

  store = pd.HDFStore("../data_rpkm/test_data.h5")
  feat_mat_df = store['rpkm']
  labels = store['labels']

  fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
    axis = 1)

  x_df = fl.drop(labels = 'labels', axis=1)
  x_test = x_df.values

  y = fl['labels']
  y_test= y.map(labeldict)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  x_train = sc.fit_transform(X = x_train)
  x_test = sc.transform(X = x_test)

  print(x_train.shape, x_test.shape)

  print(x_train, x_test)
  print(type(x_train))
  print(type(x_test))

  print("Normalized features")
  
  df = pd.DataFrame(data=x_train)
  df.to_hdf("oversampled.h5","x_train")
  df = pd.DataFrame(data=y_train)
  df.to_hdf("oversampled.h5","y_train")

  df = pd.DataFrame(data=x_test)
  df.to_hdf("oversampled.h5","x_test")
  df = pd.DataFrame(data=y_test)
  df.to_hdf("oversampled.h5","y_test")

if __name__ == '__main__': do_upsample()