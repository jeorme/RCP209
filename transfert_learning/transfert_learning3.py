# Load ResNet50 architecture & its weights
from keras import Model
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import average_precision_score

from data_gen import PascalVOCDataGenerator

model = ResNet50(include_top=True, weights='imagenet')
model.layers.pop()
# Modify top layers
x = model.layers[-1].output
x = Dense(20, activation='sigmoid', name='predictions')(x)
model = Model(input=model.input,output=x)
for i in range(len(model.layers)):
  model.layers[i].trainable = True

lr = 0.1
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['binary_accuracy'])

batch_size=32
nb_epochs=10
data_dir = "C:\\Users\\jerpetit\\Desktop\\master\\repo\\RCP209\\transfert_learning\\VOC\\VOCdevkit\\VOC2007\\" # A changer avec votre chemin
data_generator_train = PascalVOCDataGenerator('trainval', data_dir)

steps_per_epoch_train = int(len(data_generator_train.id_to_label) / batch_size) + 1
model.fit_generator(data_generator_train.flow(batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch_train,
                    epochs=nb_epochs,
                    verbose=1)

data_generator_test = PascalVOCDataGenerator('test', data_dir)
X_train = np.zeros((len(data_generator_train.images_ids_in_subset),2048))
Y_train = np.zeros((len(data_generator_train.images_ids_in_subset),20))
X_test = np.zeros((len(data_generator_test.images_ids_in_subset),2048))
Y_test = np.zeros((len(data_generator_test.images_ids_in_subset),20))
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
AP_train = np.zeros(20)
AP_test = np.zeros(20)
for c in range(20):
    AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
    AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])

print("MAP TRAIN =", AP_train.mean()*100)
print("MAP TEST =", AP_test.mean()*100)