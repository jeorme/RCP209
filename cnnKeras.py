from keras.datasets import mnist
from keras.optimizers import SGD

from kerasHelper import convertToCat

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train, y_test = convertToCat(y_train,y_test,nbCat=10)
input_shape = (28, 28, 1)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
###implement alex net :
#Une couche de convolution avec 32 filtres de taille 5×5
# suivie d’une non linéarité de type sigmoïde
# puis d’une couche de max pooling de taille 2×2.
model = Sequential()
model.add(Conv2D(32,kernel_size=(5, 5),activation='sigmoid',input_shape=(28, 28, 1),padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Une seconde couche de convolution avec 64 filtres de taille 5×5,
# suivie d’une non linéarité de type sigmoïde
# puis d’une couche de max pooling de taille 2×2.
model.add(Conv2D(64,kernel_size=(5, 5),activation='sigmoid',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#on vectorise
model.add(Flatten())
model.add(Dense(100, name='fc1',activation='sigmoid'))
model.add(Dense(10, name='fc2',activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=SGD(.5), metrics=['accuracy'])
#training
model.fit(x_train, y_train,batch_size=300, epochs=10,verbose=1)
#score
scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

