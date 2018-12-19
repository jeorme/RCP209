from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

from TD4_helper import accuracyStdMean, plotAccuracyBagging, plotAccuracyForrest, plotDepthTree, plotLearningTree, \
    plotEstimTree

digits = load_digits()
print(digits.data.shape)

# plt.gray()
# plt.matshow(digits.images[10])
# plt.show()


X = digits.data
y = digits.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
print(clf.score(X, y))

from sklearn.model_selection import train_test_split, GridSearchCV

#accuracy = []
#for i in range(200):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
#    clf = tree.DecisionTreeClassifier()
#    clf.fit(X_train, y_train)
#    Z = clf.predict(X_test)
#    accuracy.append(clf.score(X_test,y_test))
#print(np.mean(accuracy))
#print(np.std(accuracy))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=200)
clf.fit(X_train, y_train)
Z = clf.predict(X_test)
print(clf.score(X_test,y_test))

#plotAccuracyBagging(X, y, nbLoop=201)

#arbre = BaggingClassifier(tree.DecisionTreeClassifier(), n_estimators=200)
#parameters = {'max_samples' : np.random.uniform(0,1,10), 'max_features' : np.random.uniform(0,1,10)}
#clf = GridSearchCV(arbre, parameters, cv=5)
#clf.fit(X_train, y_train)
#print(clf.best_params_)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)
Z = clf.predict(X_test)
print(clf.score(X_test,y_test))

#output = []
#for i in range(200):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
#    clf = RandomForestClassifier(n_estimators=200)
#    clf.fit(X_train, y_train)
#    Z = clf.predict(X_test)
#    output.append(clf.score(X_test,y_test))
#print(np.mean(output))
#print(np.std(output))

#plotAccuracyForrest(X, y)

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5), n_estimators=200, learning_rate=2)
clf.fit(X_train, y_train)
Z = clf.predict(X_test)
print(clf.score(X_test,y_test))