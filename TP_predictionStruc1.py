from keras.datasets import mnist
from SXModel import*

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

d = 784
K = 10
learning_rate = 0.05
epochs = 20
tbatch = 100

modelXclass = SXclass(d, K, learning_rate, epochs, tbatch)
print(modelXclass)

modelXclass.fit(X_train, y_train)

pred = modelXclass.predict(X_train)
predtest = modelXclass.predict(X_test)
print("TRAINING COMPLETED... pref train=", (1.0 - modelXclass.delta(pred, y_train)) * 100.0, " pref test=", (
            1.0 - modelXclass.delta(predtest, y_test)) * 100.0)
