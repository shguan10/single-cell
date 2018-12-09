import pandas as pd
import numpy as np
import random

HP = {
  "h1_shape" : 15,
  "batch_size": 100,
  "epochs":150,
}

HPSpace = {
  "h1_shape" : [15,30,50,100,700],
  "batch_size": [10,50,100],
}

def hp2str():
  return "h1_"+str(HP["h1_shape"])+\
         "__bsize_"+str(HP["batch_size"])+\
         "__epochs_"+str(HP["epochs"])+"_"

def getXandY(filename,labeldict=None):
  # get x and y
  store = pd.HDFStore(filename)
  feat_mat_df = store['rpkm']
  #x = feat_mat_df.values
  labels = store['labels']

  fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
    axis = 1)

  x_df = fl.drop(labels = 'labels', axis=1)
  x = x_df.values

  y = fl['labels']
  if labeldict==None: 
    mapping=pd.get_dummies(y)
    labeldict={}
    for lab,vec in zip(y,mapping.values): labeldict[lab] = vec
    y=mapping
  else: y= y.map(labeldict)
  return x, y.values,labeldict

def cachedata():
  x_train, y_train,labeldict = getXandY('../data_rpkm/oversampled_train_data.h5')
  y_train = np.array(list(y_train))

  x_test, y_test, labeldict = getXandY('../data_rpkm/test_data.h5',labeldict=labeldict)
  y_test = np.array(list(y_test))

  df = pd.DataFrame(data=labeldict)
  df.to_hdf("uberon.h5","labeldict")
  
  noOfTrainingSamples, noOfFeatures = x_train.shape
  noOfClasses = y_train.shape[1]

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
  
def getdata():
  store = pd.HDFStore("oversampled.h5")
  x_train = store["x_train"].values
  y_train = store["y_train"].values
  x_test = store["x_test"].values
  y_test = store["y_test"].values
  y_test = np.array([i[0] for i in y_test])
  store.close()
  return x_train,y_train,x_test,y_test

def savehist(hist,name):
  df = pd.DataFrame(data=hist.history)
  df.to_hdf("../chkpnts/histories.h5",name)

def main():
  x_train,y_train,x_test,y_test= getdata()
  
  import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras import optimizers

  hiddenNodes = HP["h1_shape"]

  classifier = Sequential()

  classifier.add(Dense(output_dim = hiddenNodes, input_dim = x_train.shape[1],
    activation = 'tanh', init = 'glorot_uniform'))
  classifier.add(Dense(output_dim = y_train.shape[1], init = 'glorot_uniform',
    activation = 'softmax'))

  # sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
  classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

  savecb = keras.callbacks.ModelCheckpoint(filepath='../chkpnts/plainNN-'+hp2str()+'-vacc{val_acc:.3f}.hdf5', 
                                           monitor='val_acc', 
                                           verbose=1, 
                                           save_best_only=True)

  hist = classifier.fit(x_train, y_train, batch_size = HP["batch_size"], epochs = HP["epochs"],
    verbose = 1, validation_data = (x_test, y_test),callbacks=[savecb])
  
  savehist(hist,hp2str()+"hash_"+str(random.random())[2:]+"_")

if __name__ == '__main__':
  # cachedata()
  while True:
    for e in HPSpace: 
      choices = HPSpace[e]
      HP[e] = random.choice(choices)
    for _ in range(1): main()  
  # main()