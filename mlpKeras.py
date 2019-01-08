from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from kerasHelper import get_minst, convertToCat

tensorboard = TensorBoard(log_dir="_mlpMC", write_graph=False, write_images=True)

model = Sequential()

# construction d'un mlp multi couche
# 1 hidden couche : activation sigmoid
model.add(Dense(100, input_dim=784, name='fc1'))
model.add(Activation('sigmoid'))
#couche de sortie prend automatique 100 comme input (couche précédente)
model.add(Dense(10, name='fc2'))
model.add(Activation('softmax'))
print(model.summary())

##train the model : with sgd, and loss cross correlation, metrics : accuracy
learning_rate = 0.5
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
batch_size = 300
nb_epoch = 10

# convert class vectors to binary class matrices
X_train, y_train, X_test, y_test = get_minst()
Y_train, Y_test = convertToCat(y_train,y_test,nbCat=10)

model.fit(X_train, Y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1, callbacks=[tensorboard])

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))