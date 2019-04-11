import numpy as np
from SModel import *
from keras.utils import np_utils
class SXclass(SModel):
  d =0
  k=0

  def __init__(self, d, K, learning_rate=0.1, epochs=2, tbatch=100):
      SModel.__init__(self,learning_rate, epochs, tbatch)
      self.d = d
      self.K = K
      self.w = np.zeros((d, K))

  def __str__(self):
      return "SXclass - size w="+str(self.w.shape)+ " eta="+str(self.eta)+" epochs="+ str(self.epochs)+ " tbatch="+str(self.tbatch)

  def psi(self, X, Y):
      tb = X.shape[0]
      tmp = np.zeros((self.d,self.K))
      for i in range(tb):
          tmp[:,Y[i]] += X[i,:].T
      return tmp/tb

  def delta(self, y1, y2):
      delt = np.abs(y1-y2)
      delt[delt>.01]=1
      return delt.mean()

  def predict(self, X):
      pred = np.dot(X,self.w)
      return np.argmax(pred,axis=1)

  def lai(self, X, Y):
      yoneHot = np_utils.to_categorical(Y, self.K)
      pred = np.dot(X, self.w) + 1.0 - yoneHot
      return np.argmax(pred, axis=1)