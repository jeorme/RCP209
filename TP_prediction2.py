import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


outfile = 'DF_ResNet50_VOC2007.npz'
npzfile = np.load(outfile)
X_train = npzfile['X_train']
Y_train = npzfile['Y_train']
X_test = npzfile['X_test']
Y_test = npzfile['Y_test']
print("X_train=",X_train.shape, "Y_train=",Y_train.shape, " X_test=",X_test.shape, "Y_train=",Y_test.shape)


model = Sequential()
model.add(Dense(1,  input_dim=2048, name='fc1'))
model.add(Activation('sigmoid'))
model.summary()
learning_rate = 0.5/20
sgd = SGD(learning_rate)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['binary_accuracy'])

c=15 # index of 'potted plant' class
batch_size = 100
nb_epoch = 20
model.fit(X_train, Y_train[:,c],batch_size=batch_size, epochs=nb_epoch,verbose=1)

scorestrain = model.evaluate(X_train, Y_train[:,c], verbose=0)
scorestest = model.evaluate(X_test, Y_test[:,c], verbose=0)
print("perfs train - %s: %.2f%%" % (model.metrics_names[1], scorestrain[1]*100))
print("perfrs test - %s: %.2f%%" % (model.metrics_names[1], scorestest[1]*100))


LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
         'bottle', 'bus', 'car', 'cat', 'chair',
         'cow', 'diningtable', 'dog', 'horse',
         'motorbike', 'person', 'pottedplant',
         'sheep', 'sofa', 'train', 'tvmonitor']

# Computing prediction for the training set
predtrain = model.predict(X_train)
# Computing precision recall curve
precision, recall, _ = precision_recall_curve(Y_train[:,c], predtrain)
# Computing Average Precision
AP_train = average_precision_score(Y_train[:,c], predtrain)
print("Class ",c," - Average Precision  TRAIN=", AP_train*100.0)

plt.clf()
plt.plot(recall, precision, lw=2, color='navy',label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('PR curve - CLASS '+LABELS[c]+': TRAIN AUC={0:0.2f}'.format(AP_train*100.0))
plt.legend(loc="lower left")
plt.show()

# Computing prediction for the test set
predtest = model.predict(X_test)
# Computing precision recall curve
precision, recall, _ = precision_recall_curve(Y_test[:,c], predtest)
# Computing Average Precision
AP_test = average_precision_score(Y_test[:,c], predtest)
print("Class ",c," - Average Precision  TRAIN=", AP_test*100.0)

plt.clf()
plt.plot(recall, precision, lw=2, color='navy',label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('PR curve - CLASS '+LABELS[c]+': TRAIN AUC={0:0.2f}'.format(AP_test*100.0))
plt.legend(loc="lower left")
plt.show()