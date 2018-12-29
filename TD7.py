# the data, shuffled and split between train and test sets
from kerasHelper import get_minst, convertToCat
from mlpHelper import forward, computeGradient, accuracy

X_train, y_train, X_test, y_test = get_minst()

from keras.utils import np_utils

K = 10
# convert class vectors to binary class matrices
Y_train, Y_test = convertToCat(y_train,y_test,nbCat=K)
print(Y_train.shape)

import numpy as np

N = X_train.shape[0]
d = X_train.shape[1]
W = np.zeros((d, K))
b = np.zeros((1, K))
numEp = 10  # Number of epochs for gradient descent
eta = 5e-1  # Learning rate
batch_size = 300
nb_batches = int(float(N) / batch_size)
gradW = np.zeros((d, K))
gradb = np.zeros((1, K))
learningRate = 0.1

for epoch in range(numEp):
    for ex in range(nb_batches):
        # FORWARD PASS : compute prediction with current params for examples in batch
        y_predict = forward(X_train[ex * batch_size:(ex + 1) * batch_size,:], W, b)
        # BACKWARD PASS :
        # 1) compute gradients for W and b
        gradW, gradb = computeGradient(Y_train[ex * batch_size:(ex + 1) * batch_size, :], y_predict,
                                       X_train[ex * batch_size:(ex + 1) * batch_size, :], batch_size)
        # 2) update W and b parameters with gradient descent
        W = W - learningRate * gradW
        b = b - learningRate * gradb


print(accuracy(W, b, X_test, Y_test))