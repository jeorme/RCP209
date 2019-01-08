from keras.datasets import mnist
from sklearn.manifold import TSNE

from kerasHelper import loadModel, get_minst, convertToCat, convexHulls, best_ellipses, neighboring_hit, visualization

model = loadModel("cnn")
print(model.summary())

model.pop()
model.pop()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_predict = model.predict(x_test)
y_embeded = y_test[:1000]
X_embedded = TSNE(init="pca",perplexity=30,verbose=2).fit_transform(x_predict[:1000])
convex_hulls= convexHulls(X_embedded, y_embeded)
ellipses = best_ellipses(X_embedded, y_embeded)
nh = neighboring_hit(X_embedded, y_embeded)

visualization(X_embedded, y_embeded, convex_hulls, ellipses ,"CNN", nh)
