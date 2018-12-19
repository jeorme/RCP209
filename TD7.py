def forward(batch, W, b):
    """compute y predicted from W and b and a batch in the case of softmax"""
    s = np.matmul(batch, W) + b
    return softmax(s)


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1)[:, None]


def computeGradient(y_train, y_predict, x_train, batch_size):
    return np.matmul(np.transpose(x_train), (y_predict - y_train)) / batch_size, np.sum(
        y_predict - y_train) / batch_size


def accuracy(W, b, images, labels):
  pred = forward(images, W,b )
  return np.where( pred.argmax(axis=1) != labels.argmax(axis=1) , 0.,1.).mean()*100.0

from keras.datasets import mnist

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape each 28x28 image -> 784 dim. vector
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalization
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

from keras.utils import np_utils

K = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, K)
Y_test = np_utils.to_categorical(y_test, K)
print(Y_train.shape)

import numpy as np

N = X_train.shape[0]
d = X_train.shape[1]
W = np.zeros((d, K))
b = np.zeros((1, K))
numEp = 20  # Number of epochs for gradient descent
eta = 1e-1  # Learning rate
batch_size = 100
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