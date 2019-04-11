from abc import ABCMeta, abstractmethod
import numpy as np

class SModel:
  _metaclass__ = ABCMeta
  w=[]
  # Batch size
  tbatch = 10
  nbatch=0
  epochs = 10
  # Gradient step
  eta = 0.1
  # Regularization parameter
  llambda = 1e-6

  # Joint feature map Psi(x,y) in R^d
  @abstractmethod
  def psi(self, X, Y):
      pass

  # Loss-Augmented Inference (LAI):  arg max_y [<w, Psi(x,y)> + Delta(y,y*)]
  @abstractmethod
  def lai(self, X, Y):
     pass

  # Inference: arg max_y [<w, Psi(x,y)>]
  @abstractmethod
  def predict(self, X):
      pass

  @abstractmethod
  # Loss between two outputs
  def delta(self, Y1, Y2):
      pass

  def __init__(self, learning_rate=0.1, epochs=2, tbatch=100):
      self.eta=learning_rate
      self.epochs = epochs
      self.tbatch = tbatch


  def fit(self, X, Y):

      self.nbatch = int(X.shape[0]  / self.tbatch)

      for i in range(self.epochs):
          for b in range(self.nbatch):
              # Computing joint feature map psi for groud truth output
              psi2 = self.psi(X[b*self.tbatch:(b+1)*self.tbatch,:], Y[b*self.tbatch:(b+1)*self.tbatch])
              # Computing most violated constraint => LAI
              yhat = self.lai(X[b*self.tbatch:(b+1)*self.tbatch,:], Y[b*self.tbatch:(b+1)*self.tbatch])
               # Computing joint feature map psi for LAI output
              psi1 = self.psi(X[b*self.tbatch:(b+1)*self.tbatch,:], yhat)
              # Computing gradient
              grad = self.llambda*self.w + (psi1-psi2)

              self.w = self.w - 1.0*self.eta * (grad)

          pred = self.predict(X)
          print("epoch ",i, " perf train=",(1.0-self.delta(pred, Y))*100.0, " norm W=",np.sum(self.w*self.w))