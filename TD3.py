from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True, rounded=True,
            special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris_gini.pdf")

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(iris.data, iris.target)
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None,
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            filled=True, rounded=True,
            special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris_entropy.pdf")


from sklearn.model_selection import train_test_split
X_train, X_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.3)
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_train, target_train)
print(clf.score(X_train,target_train))
print(clf.score(X_test,target_test))
