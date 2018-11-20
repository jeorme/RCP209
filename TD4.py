from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
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
accuracy = clf.score(X, y)
print(accuracy)
#
# accuracy = accuracyStdMean(X, y, clf=tree.DecisionTreeClassifier())
# print("tree : ("+str(np.std(accuracy))+","+ str(np.mean(accuracy))+")")
#
# accuracy = accuracyStdMean(X, y, clf=BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5, n_estimators=200))
# print("bagging : ("+str(np.std(accuracy))+" , "+ str(np.mean(accuracy))+")")
#
# accuracy = accuracyStdMean(X, y, clf=RandomForestClassifier(n_estimators=200))
# print("tree : ("+str(np.std(accuracy))+","+ str(np.mean(accuracy))+")")

#plotAccuracyBagging(X,y)
#plotAccuracyForrest(X,y)
plotDepthTree(X,y)
plotLearningTree(X,y)
plotEstimTree(X,y)