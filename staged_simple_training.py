'''
all cell_types: 
'CL:0000353 blastoderm cell' 'UBERON:0002107 liver'
 'CL:0000057 fibroblast' 'CL:0002322 embryonic stem cell'
 'CL:0000081 blood cell' 'UBERON:0000115 lung epithelium'
 'CL:0001056 dendritic cell, human' 'CL:0000746 cardiac muscle cell'
 'UBERON:0001851 cortex' 'CL:0002034 long term hematopoietic stem cell'
 'CL:0002033 short term hematopoietic stem cell'
 'CL:0000037 hematopoietic stem cell' 'CL:1000497 kidney cell'
 'CL:0008019 mesenchymal cell' 'UBERON:0000044 dorsal root ganglion'
 'CL:0002365 medullary thymic epithelial cell' 'UBERON:0000473 testis'
 'UBERON:0000992 female gonad' 'UBERON:0000922 embryo'
 'UBERON:0002048 lung' 'CL:0000137 osteocyte'
 'UBERON:0001898 hypothalamus' 'UBERON:0001997 olfactory epithelium'
 'CL:0002321 embryonic cell' 'CL:0002319 neural cell'
 'UBERON:0004129 growth plate cartilage' 'UBERON:0001891 midbrain'
 'UBERON:0002038 substantia nigra' 'UBERON:0000007 pituitary gland'
 'CL:0000763 myeloid cell' 'CL:0000540 neuron' 'UBERON:0000045 ganglion'
 "UBERON:0001954 Ammon's horn" 'CL:0000127 astrocyte'
 'CL:0000163 endocrine cell' 'UBERON:0000955 brain'
 'UBERON:0000966 retina' 'UBERON:0002435 striatum'
 'UBERON:0010743 meningeal cluster' 'CL:0000169 type B pancreatic cell'
 'UBERON:0001264 pancreas' 'CL:0000084 T cell'
 'UBERON:0001003 skin epidermis'
 'UBERON:0001902 epithelium of small intestine' 'CL:0000235 macrophage'
 'CL:0000192 smooth muscle cell'
 '''

cl_superclass_dict = {

  'CL:0000057 fibroblast': 0,
  'CL:0000137 osteocyte': 0,

  'CL:1000497 kidney cell': 1,
  'CL:0008019 mesenchymal cell': 1,
  'CL:0002365 medullary thymic epithelial cell' : 1,

  'CL:0000163 endocrine cell': 2,
  'CL:0000169 type B pancreatic cell': 2,

  'CL:0002321 embryonic cell': 3,
  'CL:0000353 blastoderm cell': 3,
  'CL:0002322 embryonic stem cell': 3,

  'CL:0000192 smooth muscle cell': 4,
  'CL:0000746 cardiac muscle cell': 4,

  'CL:0001056 dendritic cell, human': 5,
  'CL:0000084 T cell': 5,
  'CL:0000235 macrophage': 5,

  'CL:0000081 blood cell': 6,
  'CL:0000763 myeloid cell': 6,

  'CL:0002319 neural cell': 7,
  'CL:0000540 neuron': 7,
  'CL:0000127 astrocyte': 7,

  'CL:0002034 long term hematopoietic stem cell': 8,
  'CL:0002033 short term hematopoietic stem cell': 8,
  'CL:0000037 hematopoietic stem cell': 8,
}

cl_cell_superclasses = ['connective tissue', 'other', 'endocrine',
    'embryo', 'muscle', 'leukocytes', 'haem/{stem, leuk}', 'brain',
    'haem stem']

uberon_classes = {
  "UBERON:0002048 lung":0,
  "UBERON:0000115 lung epithelium":0,

  "UBERON:0000955 brain":1,
  "UBERON:0002038 substantia nigra":1,
  "UBERON:0001954 Ammon's horn":1,
  "UBERON:0001891 midbrain":1,
  "UBERON:0002435 striatum":1,
  "UBERON:0001898 hypothalamus":1,
  "UBERON:0010743 meningeal cluster":1,

  "UBERON:0001003 skin epidermis":2,
  "UBERON:0001997 olfactory epithelium":2,
  "UBERON:0001902 epithelium of small intestine":2,

  "UBERON:0000473 testis":3,
  "UBERON:0000992 female gonad":3,
  "UBERON:0000922 embryo":3,

  "UBERON:0001264 pancreas":4,
  "UBERON:0000007 pituitary gland":4,

  "UBERON:0000045 ganglion":5,
  "UBERON:0000044 dorsal root ganglion":5,
  "UBERON:0004129 growth plate cartilage":5,

  "UBERON:0000966 retina":6,
  "UBERON:0002107 liver":6,
  "UBERON:0001851 cortex":6
}

import pandas as pd
import numpy as np

def get_cell_superclass(s):
  if s in cl_superclass_dict:
    index = cl_superclass_dict[s]
    arr = np.zeros(9)
    arr[index] = 1
    return arr
  else:
    index = uberon_classes[s]
    arr = np.zeros(7)
    arr[index] = 1
    return arr

def getXandY(filename,uberon=False,orig_labels=True,labeldict=None):
  # get x and y
  store = pd.HDFStore(filename)
  feat_mat_df = store['rpkm']
  #x = feat_mat_df.values
  labels = store['labels']

  fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
    axis = 1)

  fl = fl[uberon == fl.labels.str.contains("UBERON")]

  x_df = fl.drop(labels = 'labels', axis=1)
  x = x_df.values

  y = fl['labels']
  if orig_labels: 
    if labeldict==None: 
      mapping=pd.get_dummies(y)
      labeldict={}
      for lab,vec in zip(y,mapping.values): labeldict[lab] = vec
      y=mapping
    else: y= y.map(labeldict)
  else: y = y.map(get_cell_superclass)
  return x, y.values,labeldict

def cacheuberon(orig_labels=True):
  x_train, y_train,labeldict = getXandY('../data_rpkm/train_data.h5',uberon=True,orig_labels=orig_labels)
  y_train = np.array(list(y_train))

  x_test, y_test, labeldict = getXandY('../data_rpkm/test_data.h5',uberon=True,orig_labels=orig_labels,labeldict=labeldict)
  y_test = np.array(list(y_test))

  df = pd.DataFrame(data=labeldict)
  df.to_hdf("uberon.h5","labeldict")
  
  # COPY PASTED FROM SIMPLE_TRAINING.PY

  noOfTrainingSamples, noOfFeatures = x_train.shape
  assert(noOfFeatures == x_test.shape[1])
  noOfClasses = y_train.shape[1]
  assert(noOfClasses == y_test.shape[1])

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  x_train = sc.fit_transform(X = x_train)
  x_test = sc.transform(X = x_test)

  print(x_train.shape, x_test.shape)

  print(x_train, x_test)
  print(type(x_train))
  print(type(x_test))

  print("Normalized features")
  
  if not orig_labels:
    df = pd.DataFrame(data=x_train)
    df.to_hdf("uberon.h5","x_train")
  df = pd.DataFrame(data=y_train)
  df.to_hdf("uberon.h5","y_train2" if orig_labels else "y_train")

  if not orig_labels:
    df = pd.DataFrame(data=x_test)
    df.to_hdf("uberon.h5","x_test")
  df = pd.DataFrame(data=y_test)
  df.to_hdf("uberon.h5","y_test2" if orig_labels else "y_test")
  

def getuberon():
  store = pd.HDFStore("uberon.h5")
  x_train = store["x_train"].values
  y_train = store["y_train"].values
  y_train2 = store["y_train2"].values
  x_test = store["x_test"].values
  y_test = store["y_test"].values
  y_test2 = store["y_test2"].values
  return x_train,y_train,x_test,y_test

def main():
  x_train,y_train,y_train2,x_test,y_test,y_test2 = getuberon()

  import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras import optimizers

  hiddenNodes = 15

  classifier = Sequential()

  classifier.add(Dense(output_dim = hiddenNodes, input_dim = x_train.shape[1],
    activation = 'tanh', init = 'glorot_uniform'))
  #classifier.add(Dense(output_dim = noOfClasses, init = 'glorot_uniform',
  # activation = 'tanh'))
  classifier.add(Dense(output_dim = y_train.shape[1], init = 'glorot_uniform',
    activation = 'softmax'))

  # sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
  classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

  savecb = keras.callbacks.ModelCheckpoint(filepath='../chkpnts/uberon2-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True)

  while True:
    x = classifier.fit(x_train, y_train, batch_size = 100, epochs = 300,
      verbose = 1, validation_data = (x_test, y_test), callbacks=[savecb])
    session = keras.backend.get_session()
    initial_weights = classifier.get_weights()
    new_weights = [keras.initializers.glorot_uniform()(w.shape).eval(session=session) for w in initial_weights]
    classifier.set_weights(new_weights)

  loss = classifier.evaluate(x_test, y_test,
    verbose = 1)

  # "Evaluated test samples"

  print(loss)
  print(classifier.metrics_names)

  y_pred = classifier.predict(x_test)

  from sklearn import metrics
  matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
  np.set_printoptions(threshold = np.nan)
  print(matrix)

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

  for x in matrix:
    print(x)
  '''
  d_new = {}
  for i in range(len(d)):
    d_new[i] = np.argmax(d[i])
  print(d_new)
  '''


if __name__ == '__main__':
  cacheuberon()
  # main()
