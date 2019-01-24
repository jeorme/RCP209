from kerasHelper import convertToCat, get_minst, perceptron1couche

#comparaison sigmoid / relu
X_train, y_train, X_test, y_test = get_minst()
Y_train, Y_test = convertToCat(y_train, y_test, nbCat=10)
batch_size = 300
nb_epoch = 10
perceptron1couche(X_train , Y_train, X_test, Y_test, batch_size, nb_epoch, activation1 = "sigmoid",activation2 = "softmax", isSave=False)
perceptron1couche(X_train , Y_train, X_test, Y_test, batch_size, nb_epoch, activation1 = "relu",activation2 = "softmax", isSave=False)

