import keras
import keras.backend as K
import keras.models as KM
import keras.layers as KL
import keras.engine as KE
from keras import optimizers

import pandas as pd
import numpy as np

import random
from dim_red_models import PCA
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

combined_dict = {

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

  "UBERON:0002048 lung":9,
  "UBERON:0000115 lung epithelium":9,

  "UBERON:0000955 brain":10,
  "UBERON:0002038 substantia nigra":10,
  "UBERON:0001891 midbrain":10,

  "UBERON:0001954 Ammon's horn":11,
  "UBERON:0001898 hypothalamus":11,

  "UBERON:0002435 striatum":12,
  "UBERON:0010743 meningeal cluster":12,


  "UBERON:0001003 skin epidermis":13,
  "UBERON:0001997 olfactory epithelium":13,
  "UBERON:0001902 epithelium of small intestine":13,

  "UBERON:0000473 testis":14,
  "UBERON:0000992 female gonad":14,
  "UBERON:0000922 embryo":14,

  "UBERON:0001264 pancreas":15,
  "UBERON:0000007 pituitary gland":15,

  "UBERON:0000045 ganglion":16,
  "UBERON:0000044 dorsal root ganglion":16,
  "UBERON:0004129 growth plate cartilage":16,

  "UBERON:0000966 retina":17,
  "UBERON:0002107 liver":17,
  "UBERON:0001851 cortex":17
}

HP = {
  "h1_shape" : 100,
  "h2_shape" : 50,
  "batch_size": 100,
  "loss_weights":[0,1.],
  "epochs":50,
  "pretrained":True,
  "whichtrain":"PCAcombined",
  "stage":2
}

HPSpace = {
  "h1_shape" : [15],
  "h2_shape" : [15],
  "batch_size": [50],
  "loss_weights":[0],
  "pretrained":[True]
}

def hp2str():
  return HP["whichtrain"]+str(HP["stage"])+"__"+\
         "h1_"+str(HP["h1_shape"])+\
         "__h2_"+str(HP["h2_shape"])+\
         "__bsize_"+str(HP["batch_size"])+\
         "__lcoeffs_"+str(HP["loss_weights"])+"_"+str(1.)+\
         "__epochs_"+str(HP["epochs"])+\
         "__ptrain_"+str(HP["pretrained"])+"_"

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

def cacheover(allf="oversampled.h5",uberon=False):
  store = pd.HDFStore(allf)
  x_train = store["x_train"]
  y_train = store["y_train"]
  x_test = store["x_test"]
  y_test = store["y_test"]

  store.close()

  def process(xtraindf,ytraindf,uberon=True,orig_labels=True,labeldict=None):
    fl = pd.concat([xtraindf, pd.DataFrame(ytraindf.rename('labels'))],
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
  
  fname = "overub.h5" if uberon else "overcl.h5"
  xtrain,ytrain,labeldict = process(x_train,y_train,uberon=uberon,orig_labels=False)

  xtrain2,ytrain2,labeldict = process(x_train,y_train,uberon=uberon,orig_labels=True,labeldict=labeldict)

  xtest,ytest,labeldict = process(x_test,y_test,labeldict=labeldict,uberon=uberon,orig_labels=False)
  
  xtest2,ytest2,labeldict = process(x_test,y_test,uberon=uberon,orig_labels=True,labeldict=labeldict)  
  
  df = pd.DataFrame(data=labeldict)
  df.to_hdf(fname,"labeldict")


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
    df.to_hdf(fname,"x_train")
  df = pd.DataFrame(data=y_train)
  df.to_hdf(fname,"y_train2" if orig_labels else "y_train")

  if not orig_labels:
    df = pd.DataFrame(data=x_test)
    df.to_hdf(fname,"x_test")
  df = pd.DataFrame(data=y_test)
  df.to_hdf(fname,"y_test2" if orig_labels else "y_test")

def cachedata(orig_labels=True,uberon=True,trainf='../data_rpkm/train_data.h5',testf='../data_rpkm/test_data.h5'):
  fname = "uberon.h5" if uberon else "cl.h5"
  x_train, y_train,labeldict = getXandY(trainf,uberon=uberon,orig_labels=orig_labels)
  y_train = np.array(list(y_train))

  x_test, y_test, labeldict = getXandY(testf,uberon=uberon,orig_labels=orig_labels,labeldict=labeldict)
  y_test = np.array(list(y_test))

  df = pd.DataFrame(data=labeldict)
  df.to_hdf(fname,"labeldict")
  
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
  
  if not orig_labels:
    df = pd.DataFrame(data=x_train)
    df.to_hdf(fname,"x_train")
  df = pd.DataFrame(data=y_train)
  df.to_hdf(fname,"y_train2" if orig_labels else "y_train")

  if not orig_labels:
    df = pd.DataFrame(data=x_test)
    df.to_hdf(fname,"x_test")
  df = pd.DataFrame(data=y_test)
  df.to_hdf(fname,"y_test2" if orig_labels else "y_test")

def cachecl():
  cachedata(orig_labels=False,uberon=False)
  cachedata(orig_labels=True,uberon=False)

def get_cell_superclass_combined(label):
  num = combined_dict[label]
  vec = np.zeros(18)
  vec[num] = 1
  return vec

def normalizeall():
  # GET TRAIN
  store = pd.HDFStore("../data_rpkm/train_data.h5")
  feat_mat_df = store['rpkm']
  labels = store['labels']
  store.close()

  fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
                  axis = 1)

  x_df = fl.drop(labels = 'labels', axis=1)
  x_train = x_df.values

  y=fl['labels']
  mapping=pd.get_dummies(y)
  labeldict={}
  for lab,vec in zip(y,mapping.values): labeldict[lab] = vec
  y_train2=mapping.values
  y_train = y.map(get_cell_superclass_combined)

  # GET TEST
  store = pd.HDFStore("../data_rpkm/test_data.h5")
  feat_mat_df = store['rpkm']
  labels = store['labels']
  store.close()

  fl = pd.concat([feat_mat_df, pd.DataFrame(labels.rename('labels'))],
                  axis = 1)
  y=fl['labels']
  x_df = fl.drop(labels = 'labels', axis=1)
  x_test = x_df.values

  y_test2=y.map(labeldict)
  y_test = y.map(get_cell_superclass_combined)

  y_train = np.array(list(y_train))
  y_test = np.array(list(y_test))
  y_test2 = np.array(list(y_test2))
  print(y_train.shape,y_train2.shape,y_test.shape,y_test2.shape)

  # normalize x_train and x_test
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  x_train = sc.fit_transform(X = x_train)
  x_test = sc.transform(X = x_test)
  print("normalized data")
  
  # WRITE TO H5
  fname = "../data_rpkm/allnormalized.h5"
  df = pd.DataFrame(data=x_train)
  df.to_hdf(fname,"x_train")
  df = pd.DataFrame(data=y_train)
  df.to_hdf(fname,"y_train")
  df = pd.DataFrame(data=y_train2)
  df.to_hdf(fname,"y_train2")

  df = pd.DataFrame(data=x_test)
  df.to_hdf(fname,"x_test")
  df = pd.DataFrame(data=y_test)
  df.to_hdf(fname,"y_test")
  df = pd.DataFrame(data=y_test2)
  df.to_hdf(fname,"y_test2")

  df = pd.DataFrame(data=labeldict)
  df.to_hdf(fname,"labeldict")
  
def getalldata():
  fname = "../data_rpkm/allnormalized.h5"
  store = pd.HDFStore(fname)
  x_train = store["x_train"].values
  y_train = store["y_train"].values
  y_train2 = store["y_train2"].values
  x_test = store["x_test"].values
  y_test = store["y_test"].values
  y_test2 = store["y_test2"].values
  store.close()
  return x_train,y_train,y_train2,x_test,y_test,y_test2

def getdata(uberon=True):
  fname = "uberon.h5" if uberon else "cl.h5"
  store = pd.HDFStore(fname)
  x_train = store["x_train"].values
  y_train = store["y_train"].values
  y_train2 = store["y_train2"].values
  x_test = store["x_test"].values
  y_test = store["y_test"].values
  y_test2 = store["y_test2"].values
  store.close()
  return x_train,y_train,y_train2,x_test,y_test,y_test2

class StagedNN:
  """
  Staged NN for gene expression classification
  """
  def __init__(self,shapes):
    """
    shapes = (input_shape,h1_shape,o1_shape,h2_shape,o2_shape)
    """
    self.layershapes = shapes

  def build(self,pretrained=False):
    input_shape,h1_shape,o1_shape,h2_shape,o2_shape = self.layershapes
    input_exp = KL.Input(shape=(input_shape,), name="input_expressions")
    h1 = KL.Dense(h1_shape, 
                  name="h1", 
                  activation="tanh", 
                  kernel_initializer="glorot_uniform")(input_exp)
    self.o1 = KL.Dense(o1_shape,
                       name="o1",
                       activation = 'softmax',
                       kernel_initializer = 'glorot_uniform')(h1)

    h2a = KL.Dense(h2_shape,activation = 'tanh', kernel_initializer = 'glorot_uniform')(self.o1)
    h2b = KL.Dense(h2_shape,activation = 'tanh', kernel_initializer = 'glorot_uniform')(h1)
    h2 = KL.Add()([h2a,h2b])
    self.o2 = KL.Dense(o2_shape, 
                       name="o2", 
                       activation = 'softmax',
                       kernel_initializer = 'glorot_uniform')(h2)
    self.stage = KM.Model(inputs=input_exp,outputs=self.o1, name="staged_nn_stage")
    self.model = KM.Model(inputs=input_exp,outputs=[self.o1,self.o2], name="staged_nn")

    if pretrained: self.model.load_weights("../chkpnts/PCAcombined1__h1_15__h2_50__bsize_50__lcoeffs_0_1.0__epochs_50__ptrain_True_-vacc0.640.hdf5",by_name=True)
    # self.model.load_weights("../chkpnts/uberon2-0.48.hdf5",by_name=True)

    self.stage.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])

    self.model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy',
                        metrics = ['accuracy'], loss_weights=[HP["loss_weights"],1.])
  def trainstage(self,train_vectors,test_vectors,callbacks=[]):
    x_train,y_train = train_vectors
    x_test,y_test = test_vectors
    return self.stage.fit(x_train, 
                          y_train, 
                          batch_size = HP["batch_size"], 
                          epochs = HP["epochs"],
                          verbose = 1, 
                          validation_data = (x_test, y_test), 
                          callbacks=callbacks)

  def train(self,train_vectors,test_vectors,callbacks=[]):
    x_train,y_train,y_train2 = train_vectors
    x_test,y_test,y_test2 = test_vectors
    return self.model.fit(x_train, 
                          [y_train,y_train2], 
                          batch_size = HP["batch_size"], 
                          epochs = HP["epochs"],
                          verbose = 1, 
                          validation_data = (x_test, [y_test,y_test2]), 
                          callbacks=callbacks)

def pretrain():
  x_train,y_train,y_train2,x_test,y_test,y_test2 = getalldata()
  x_train, x_test = PCA(x_train, x_test)
  network = StagedNN((x_train.shape[1],
                      HP["h1_shape"],
                      y_train.shape[1],
                      HP["h2_shape"],
                      y_train2.shape[1]))
  network.build(pretrained=False)
  savecb = keras.callbacks.ModelCheckpoint(filepath='../chkpnts/'+hp2str()+'-vacc{val_acc:.3f}.hdf5', 
                                           monitor='val_acc', 
                                           verbose=1, 
                                           save_best_only=True)

  hist = network.trainstage((x_train,y_train),
                             (x_test,y_test),
                             callbacks=[savecb])

  savehist(hist,hp2str()+"hash_"+str(random.random())[2:]+"_")

def savehist(hist,name):
  df = pd.DataFrame(data=hist.history)
  df.to_hdf("../chkpnts/histories.h5",name)


def main():
  x_train,y_train,y_train2,x_test,y_test,y_test2 = getalldata()
  x_train, x_test = PCA(x_train, x_test)

  network = StagedNN((x_train.shape[1],
                      HP["h1_shape"],
                      y_train.shape[1],
                      HP["h2_shape"],
                      y_train2.shape[1]))
  network.build(pretrained=HP["pretrained"])

  # sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
  savecb = keras.callbacks.ModelCheckpoint(filepath='../chkpnts/'+hp2str()+'-vo2acc{val_o2_acc:.3f}.hdf5', 
                                           monitor='val_o2_acc', 
                                           verbose=1, 
                                           save_best_only=True)

  hist = network.train((x_train,y_train,y_train2),
                       (x_test,y_test,y_test2),
                       callbacks=[savecb])

  savehist(hist,hp2str()+"hash_"+str(random.random())[2:]+"_")
  # while True:
  #   session = K.get_session()
  #   initial_weights = classifier.get_weights()
  #   new_weights = [keras.initializers.glorot_uniform()(w.shape).eval(session=session) for w in initial_weights]
  #   classifier.set_weights(new_weights)

  _,y_pred = network.model.predict(x_test)

  from sklearn import metrics
  matrix = metrics.confusion_matrix(y_test2.argmax(axis=1), y_pred.argmax(axis=1))
  np.set_printoptions(threshold = np.nan)
  print(matrix)

  matrix = []
  y_pred = np.argmax(y_pred, axis = 1)
  y_test = np.argmax(y_test2, axis = 1)
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

  for x in matrix: print(x)

def labellayers():
  x_train,y_train,y_train2,x_test,y_test,y_test2 = getuberon()
  model = KM.Sequential()
  model.add(KL.Dense(15, name="h1", input_dim = x_train.shape[1],
    activation = 'tanh', init = 'glorot_uniform'))
  model.add(KL.Dense(output_dim = y_train.shape[1], name="o1", init = 'glorot_uniform',
    activation = 'softmax'))
  model.load_weights("../chkpnts/uberon-0.63.hdf5")
  model.save_weights("../chkpnts/labeled_uberon-0.63.hdf5")

if __name__ == '__main__':
  # cachecl()
  # normalizeall()
  # pretrain()
  while True:
    for e in HPSpace: 
      choices = HPSpace[e]
      HP[e] = random.choice(choices)
    for _ in range(1): main()      
  # labellayers()
