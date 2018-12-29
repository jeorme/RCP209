from keras.datasets import mnist
from keras.utils import np_utils

def get_minst(isPrint=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape each 28x28 image -> 784 dim. vector
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # Normalization
    X_train /= 255
    X_test /= 255
    if isPrint:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
    return X_train, y_train, X_test, y_test

def convertToCat(y_train,y_test,nbCat):
    return np_utils.to_categorical(y_train, nbCat), np_utils.to_categorical(y_test, nbCat)