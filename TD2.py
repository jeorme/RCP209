# importations
import numpy as np    # si pas encore fait
import matplotlib.pyplot as plt

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

cmp = np.array(['r','g'])
plt.scatter(X_train[:,0],X_train[:,1],c=cmp[y_train],s=50,edgecolors='none')
plt.show()
plt.scatter(X_test[:,0],X_test[:,1],c='none',s=50,edgecolors=cmp[y_test])
plt.show()

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1)
# KFold pour différentes valeurs de k
from sklearn.model_selection import KFold
# valeurs de k
kcvfs=np.array([2, 3, 5, 7, 10, 13, 16, 20,400])
# préparation des listes pour stocker les résultats
kcvscores = list()
kcvscores_std = list()
test = list()

for kcvf in kcvfs:    # pour chaque valeur de k
  kf = KFold(n_splits=kcvf)
  these_scores = list()
  test_scores = list()
  # apprentissage puis évaluation d'un modèle sur chaque split
  for train_idx, test_idx in kf.split(X_train):
    clf.fit(X_train[train_idx], y_train[train_idx])
    these_scores.append(clf.score(X_train[test_idx], y_train[test_idx]))
  # calcul de la moyenne et de l'écart-type des performances obtenues
  kcvscores.append(np.mean(these_scores))
  kcvscores_std.append(np.std(these_scores))
  test.append(clf.score(X_test[:], y_test[:]))
  print("k fold : "+str(kcvf)+" done")

# création de np.array à partir des listes
kcvscores, kcvscores_std,test_plot = np.array(kcvscores), np.array(kcvscores_std), np.array(test)
plt.plot(kcvfs, kcvscores, 'b')
plt.show()
plt.plot(kcvfs, test_plot, 'b')
plt.show()
plt.plot(kcvfs, kcvscores+kcvscores_std, 'b--')
plt.show()
plt.plot(kcvfs, kcvscores-kcvscores_std, 'b--')
plt.show()
