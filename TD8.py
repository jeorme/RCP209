from keras.models import Sequential
from keras.optimizers import SGD

from kerasHelper import get_minst, convertToCat
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="_mnist", write_graph=False, write_images=True)

model = Sequential()
from keras.layers import Dense, Activation

model.add(Dense(10, input_dim=784, name='fc1'))
model.add(Activation('softmax'))
print(model.summary())
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


