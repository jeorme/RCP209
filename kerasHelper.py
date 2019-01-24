import matplotlib
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import linalg
from scipy.spatial.qhull import ConvexHull
from sklearn.mixture import GaussianMixture
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


def convertToCat(y_train, y_test, nbCat):
    return np_utils.to_categorical(y_train, nbCat), np_utils.to_categorical(y_test, nbCat)


def loadModel(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded ")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
    return model


from keras.models import model_from_yaml


def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")


def convexHulls(points, labels):
    # computing convex hulls for a set of points with asscoiated labels
    convex_hulls = []
    for i in range(10):
        convex_hulls.append(ConvexHull(points[labels == i, :]))
    return convex_hulls


def best_ellipses(points, labels):
    # computing best fitting ellipse for a set of points with associated labels
    gaussians = []
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1, covariance_type='full').fit(points[labels == i, :]))
    return gaussians


from sklearn.neighbors import NearestNeighbors


def neighboring_hit(points, labels):
    k = 6
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(len(points)):
        tx = 0.0
        for j in range(1, k + 1):
            if (labels[indices[i, j]] == labels[i]):
                tx += 1
            tx /= k
        txsc[labels[i]] += tx
        nppts[labels[i]] += 1
        txs += tx

    for i in range(10):
        txsc[i] /= nppts[i]
    print(txsc)

    return txs / len(points)


def visualization(points2D, labels, convex_hulls, ellipses, projname, nh):
    points2D_c = []
    for i in range(10):
        points2D_c.append(points2D[labels == i, :])
    # Data Visualization
    cmap = matplotlib.cm.get_cmap("jet", 10)

    plt.figure(figsize=(3.841, 7.195), dpi=100)
    plt.set_cmap(cmap)
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(311)
    plt.scatter(points2D[:, 0], points2D[:, 1], c=labels, s=3, edgecolors='none', cmap=cmap, alpha=1.0)
    plt.colorbar(ticks=range(10))

    plt.title("2D " + projname + " -plt NH=" + str(nh * 100.0))

    vals = [i / 10.0 for i in range(10)]
    sp2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices, convex_hulls[i].vertices[0])
        sp2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1], '-', label='$%i$' % i, color=cmap(vals[i]))
    plt.colorbar(ticks=range(10))
    plt.title(projname + " Convex Hulls")

    def plot_results(X, Y_, means, covariances, index, title, color):
        splot = plt.subplot(3, 1, 3)
        for i, (mean, covar) in enumerate(zip(means, covariances)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color, alpha=0.2)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.6)
            splot.add_artist(ell)

        plt.title(title)

    plt.subplot(313)

    for i in range(10):
        plot_results(points2D[labels == i, :], ellipses[i].predict(points2D[labels == i, :]), ellipses[i].means_,
                     ellipses[i].covariances_, 0, projname + " fitting ellipses", cmap(vals[i]))

    plt.savefig(projname + ".png", dpi=100)
    plt.show()


def perceptron1couche(X_train, Y_train, X_test, Y_test, batch_size, nb_epoch, learning_rate=.5, activation1="sigmoid",
                      activation2="softmax", isSave=False):
    # perceptron a 1 couche
    model = Sequential()

    # construction d'un mlp multi couche
    # 1 hidden couche : activation sigmoid
    model.add(Dense(100, input_dim=784, name='fc1'))
    model.add(Activation(activation1))
    # couche de sortie prend automatique 100 comme input (couche précédente)
    model.add(Dense(10, name='fc2'))
    model.add(Activation(activation2))
    print(model.summary())
    ##train the model : with sgd, and loss cross correlation, metrics : accuracy
    sgd = SGD(learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)
    if isSave:
        saveModel(model, "mlp")
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
