from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from kerasHelper import get_minst, convertToCat, saveModel, perceptron1couche

tensorboard = TensorBoard(log_dir="_mlpMC", write_graph=False, write_images=True)

# convert class vectors to binary class matrices
X_train, y_train, X_test, y_test = get_minst()
Y_train, Y_test = convertToCat(y_train, y_test, nbCat=10)
batch_size = 300
nb_epoch = 10

perceptron1couche(X_train , Y_train, X_test, Y_test, batch_size, nb_epoch, activation1 = "sigmoid",activation2 = "softmax", isSave=False)