from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn import tree


def accuracyStdMean(X, y, clf, nbloop = 100):
    accuracy = []
    for i in range(nbloop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    return accuracy

def plotAccuracyBagging(X,y,nbLoop=201):
    accuracy = []
    for i in range(1, nbLoop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf = BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=i)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    plt.scatter(range(1, 201), accuracy)
    plt.show()

def plotAccuracyForrest(X,y,nbLoop=201):
    accuracy = []
    for i in range(1, nbLoop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf = RandomForestClassifier( n_estimators=i)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    plt.scatter(range(1, 201), accuracy)
    plt.show()

def plotDepthTree(X,y,maxDepth=21):
    accuracy = []
    for i in range(1, maxDepth):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=i), n_estimators=200,
                                 learning_rate=2)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    plt.scatter(range(1, 21), accuracy)
    plt.show()

def plotLearningTree(X,y,learningRate=21):
    accuracy = []
    for i in range(1, learningRate):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5), n_estimators=200,
                                 learning_rate=learningRate)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    plt.scatter(range(1, 21), accuracy)
    plt.show()

def plotEstimTree(X,y,estimator=201):
    accuracy = []
    for i in range(1, estimator):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90)
        clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5), n_estimators=i,
                                 learning_rate=1)
        clf.fit(X_train, y_train)
        accuracy.append(clf.score(X_test, y_test))
    plt.scatter(range(1, 21), accuracy)
    plt.show()