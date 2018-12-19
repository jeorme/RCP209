# importations
import numpy as np
import matplotlib.pyplot as plt
# mode interactif
plt.ion()    # si ce n'est déjà fait

# définir matrices de rotation et de dilatation
rot = np.array([[0.94, -0.34], [0.34, 0.94]])
sca = np.array([[3.4, 0], [0, 2]])

# générer données classe 1
np.random.seed(150)
c1d = (np.random.randn(400,2)).dot(sca).dot(rot)

# générer données classe 2
c2d1 = np.random.randn(100,2)+[-10, 2]
c2d2 = np.random.randn(100,2)+[-7, -2]
c2d3 = np.random.randn(100,2)+[-2, -6]
c2d4 = np.random.randn(100,2)+[5, -7]
data = np.concatenate((c1d, c2d1, c2d2, c2d3, c2d4))

# générer étiquettes de classe
l1c = np.ones(400, dtype=int)
l2c = np.zeros(400, dtype=int)
labels = np.concatenate((l1c, l2c))

# découpage initial en données d'apprentissage et données de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)

# affichage des données d'apprentissage et de test
cmp = np.array(['r','g'])
fig = plt.figure()
plt.scatter(X_train[:,0],X_train[:,1],c=cmp[y_train],s=50,edgecolors='none')
plt.scatter(X_test[:,0],X_test[:,1],c='none',s=50,edgecolors=cmp[y_test])
plt.show()
# emploi de PMC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'hidden_layer_sizes':[(5,), (20,), (50,), (100,), (150,), (200,)],
                    'alpha':   [0.001, 0.01, 1, 2]}
clf = GridSearchCV(MLPClassifier(solver='lbfgs'), tuned_parameters, cv=5)

# exécution de grid search
clf.fit(X_train, y_train)
print(clf.best_params_)

n_hidden = np.array([5, 20, 50, 100, 150, 200])
alphas = np.array([0.001, 0.01, 1, 2])
xx, yy = np.meshgrid(n_hidden, alphas)
Z = clf.cv_results_['mean_test_score'].reshape(xx.shape)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("Neurones cachés")
ax.set_ylabel("alpha")
ax.set_zlabel("Taux de bon classement")
ax.plot_wireframe(xx, yy, Z)
plt.show()

print(clf.cv_results_)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


from sklearn.model_selection import RandomizedSearchCV
param = {'hidden_layer_sizes': np.random.randint(5,200,20),
                    'alpha':   np.random.uniform(0,2,20)}
rd = RandomizedSearchCV(MLPClassifier(solver='lbfgs'), n_iter=40, param_distributions=param, cv=5)
rd.fit(X_train, y_train)
print(rd.best_params_)
print(rd.score(X_train,y_train))
print(rd.score(X_test,y_test))