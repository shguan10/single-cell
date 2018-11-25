import pandas as pd
import numpy as np

def normalize_rows(x):
  return x / (x.sum(axis=1)[:, np.newaxis])

def make_np_array(y):
  # L = []
  # for i in y:
  #   L.append(i)
  # L = np.array(L)
  return np.array([i for i in y])

def getOneHotEncoding(labels):
  L = labels.unique()
  # d = {}
  # for i in range(len(L)):
  #   one_hot = [0] * len(L)
  #   one_hot[i] = 1
  #   d[L[i]] = np.array(one_hot)
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

  # x_train, y_train, d = getXandYandDict(DATA_DIR+'train_data.h5')
  x_train, y_train, d = getXandYandDict(DATA_DIR+'train_data.h5')

  # print("Normalize training by row")
  # x_train = normalize_rows(x_train)
  # print(x_train.sum(axis=1))
  y_train = make_np_array(y_train)

  x_test, y_test = getXandY(DATA_DIR+'test_data.h5', d)
  y_test = make_np_array(y_test)
  # import pdb
  # pdb.set_trace()

  # print("Normalize test by row")
  # x_test = normalize_rows(x_test)
  # print(x_test.sum(axis=1))
  # y_test = make_np_array(y_test)
  # from sklearn.model_selection import train_test_split
  # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,test_size = 0.2)

  noOfTrainingSamples, noOfFeatures = x_train.shape
  assert(noOfFeatures == x_test.shape[1])
  noOfClasses = y_train.shape[1]
  assert(noOfClasses == y_test.shape[1])

  print("Split data")
  # pre-processing and NN code based on example in
  # https://medium.com/@pushkarmandot/
  # build-your-first-deep-learning-neural-network-model-using-keras-in-
  # python-a90b5864116d

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  x_train = sc.fit_transform(X = x_train)
  x_test = sc.transform(X = x_test)

  import pickle as pk
  # with open("temp.pk","wb") as f: pk.dump((x_train,x_test),f)
  with open("temp.pk","rb") as f: (x_train,x_test) = pk.load(f)

  print(x_train.shape, x_test.shape)
  print(y_train.shape, y_test.shape)

  print("Normalized features")
  
  import keras
  from keras.models import Sequential
  from keras.layers import Dense

  hiddenNodes = 796

  classifier = Sequential()

  classifier.add(Dense(output_dim = hiddenNodes, input_dim = noOfFeatures,
    activation = 'tanh', init = 'glorot_uniform'))
  # classifier.add(Dense(output_dim = hiddenNodes, input_dim = hiddenNodes,
  #   activation = 'tanh', init = 'glorot_uniform'))
  # classifier.add(Dense(output_dim = hiddenNodes, input_dim = hiddenNodes,
    # activation = 'tanh', init = 'glorot_uniform'))
  classifier.add(Dense(output_dim = noOfClasses, input_dim = hiddenNodes, init = 'glorot_uniform',
    activation = 'softmax'))
  classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

  save_cb = keras.callbacks.ModelCheckpoint("../chkpnts/weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                              monitor="val_acc", period=5)

  history = classifier.fit(x_train, y_train, batch_size = 10, epochs = 100, verbose = 1, callbacks=[save_cb],
                            shuffle=1, validation_data = (x_test, y_test))
  import matplotlib.pyplot as plt
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

if __name__=="__main__": main()

