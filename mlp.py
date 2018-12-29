from keras.datasets import mnist

# the data, shuffled and split between train and test sets
from kerasHelper import get_minst, convertToCat
from mlpHelper import forward, computeGradient, accuracy, computeGradientHidden, forwardMLP, accuracyMLP

# convert class vectors to binary class matrices
K = 10
X_train, y_train, X_test, y_test = get_minst()
# convert class vectors to binary class matrices
Y_train, Y_test = convertToCat(y_train,y_test,nbCat=K)

import numpy as np

L = 100
N = X_train.shape[0]
d = X_train.shape[1]
Wh = np.zeros((d, L))
Wy = np.zeros((L, K))
bh = np.zeros((1, L))
by = np.zeros((1, K))

numEp = 20  # Number of epochs for gradient descent
eta = 1e-1  # Learning rate
batch_size = 100
nb_batches = int(float(N) / batch_size)
gradWh = np.random.normal((d, L),scale=.1)
gradWy = np.random.normal((L, K),scale=.1)
gradbh = np.random.normal((1, L),scale=.1)
gradby = np.random.normal((1, K),scale=.1)
learningRate = 0.1

for epoch in range(numEp):
    for ex in range(nb_batches):
        # FORWARD PASS : compute prediction with current params for examples in batch
        hidden, y_predict = forwardMLP(X_train[ex * batch_size:(ex + 1) * batch_size, :], Wh, bh, Wy, by)
        # BACKWARD PASS :
        # 1) compute gradients for W and b
        gradWy, gradby = computeGradient(Y_train[ex * batch_size:(ex + 1) * batch_size, :], y_predict,
                                       hidden, batch_size)
        gradWh, gradbh = computeGradientHidden(Y_train[ex * batch_size:(ex + 1) * batch_size, :], y_predict, Wy,
                                               X_train[ex * batch_size:(ex + 1) * batch_size, :]
,        hidden, batch_size)
        #gradient for hidden part
        # 2) update W and b parameters with gradient descent
        Wy= Wy - learningRate * gradWy
        by = by - learningRate * gradby
        Wh = Wh - learningRate * gradWh
        bh = bh - learningRate * gradbh

print(accuracyMLP(Wh, bh, Wy, by, X_test, Y_test))
