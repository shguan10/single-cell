import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
import pickle as pk


def normalize_rows(x):
  return x / (x.sum(axis=1)[:, np.newaxis])

def make_np_array(y):
  return np.array([i for i in y])

def getOneHotEncoding(labels):
  L = labels.unique()
  return {L[i]: [j==i for j in range(len(L))] for i in range(len(L)) }

def getXandYandDict(filename):
  # get x and y
  store = pd.HDFStore(filename)
  feat_mat_df = store['rpkm']
  x = feat_mat_df.values
  labels = store['labels']
  d = getOneHotEncoding(labels)
  y = labels.apply(lambda s: d[s]).values
  assert(x.shape[0] == y.shape[0])
  return x, y, d

def getXandY(filename, d):
  # get x and y
  store = pd.HDFStore(filename)
  feat_mat_df = store['rpkm']
  x = feat_mat_df.values
  labels = store['labels']
  y = labels.apply(lambda s: d[s]).values
  assert(x.shape[0] == y.shape[0])
  return x, y

def main():
  DATA_DIR = "../data_rpkm/"

  x_train, y_train, d = getXandYandDict(DATA_DIR+'train_data.h5')
  # x_train, y_train, d = getXandYandDict(DATA_DIR+'all_data.h5')

  y_train = make_np_array(y_train)

  x_test, y_test = getXandY(DATA_DIR+'test_data.h5', d)
  y_test = make_np_array(y_test)

  noOfTrainingSamples, noOfFeatures = x_train.shape
  assert(noOfFeatures == x_test.shape[1])
  noOfClasses = y_train.shape[1]
  assert(noOfClasses == y_test.shape[1])

  print("Split data")
  # pre-processing and NN code based on example in
  # https://medium.com/@pushkarmandot/build-your-first-deep-learning-neural-network-model-using-keras-in-python-a90b5864116d

  sc = StandardScaler()
  x_train = sc.fit_transform(X = x_train)
  x_test = sc.transform(X = x_test)

  # with open("temp.pk","wb") as f: pk.dump((x_train,x_test),f)
  # with open("temp.pk","rb") as f: (x_train,x_test) = pk.load(f)

  print(x_train.shape, x_test.shape)
  print(y_train.shape, y_test.shape)

  print("Normalized features")
  
  hiddenNodes = 796

  classifier = Sequential()

  classifier.add(Dense(output_dim = hiddenNodes, input_dim = noOfFeatures,
  	
    activation = 'tanh', init = 'glorot_uniform'))
  classifier.add(Dense(output_dim = noOfFeatures, input_dim = hiddenNodes, init = 'glorot_uniform',
    activation = 'tanh'))
  sgd = keras.optimizers.SGD(lr=100, momentum=0.0, decay=0.0, nesterov=False)
  classifier.compile(optimizer = sgd, loss = keras.losses.mean_squared_error)

  save_cb = keras.callbacks.ModelCheckpoint("../chkpnts/daeweights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                              monitor="val_loss", period=5)

  history = classifier.fit(x_train, x_train, batch_size = 10, epochs = 100, verbose = 1, callbacks=[save_cb],
                            shuffle=1, validation_data = (x_test, x_test))
  import matplotlib.pyplot as plt
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

if __name__=="__main__": main()