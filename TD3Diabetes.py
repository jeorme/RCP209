import pydotplus
from sklearn.datasets import load_diabetes
from sklearn import tree
from sklearn.model_selection import train_test_split
diabetes = load_diabetes()
X_train,X_test,y_train,y_test = train_test_split(diabetes.data,diabetes.target,test_size=.3)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
dot_data = tree.export_graphviz(clf, out_file=None,
            feature_names=diabetes.feature_names,
            class_names=str(diabetes.target),
            filled=True, rounded=True,
            special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("diabetes.pdf")
print((clf.score(X_train,y_train),clf.score(X_test,y_test)))
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'max_depth':[3,4,6,7,8,9,10]}
clfCV = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5)
clfCV.fit(X_train,y_train)
print(clfCV.best_params_)
print((clfCV.score(X_train,y_train),clfCV.score(X_test,y_test)))
#dot_data_CV = tree.export_graphviz(clfCV, out_file=None,
#            feature_names=diabetes.feature_names,
#            class_names=str(diabetes.target),
#            filled=True, rounded=True,
#            special_characters=True)
#graphCV = pydotplus.graph_from_dot_data(dot_data_CV)
#graphCV.write_pdf("diabetesCV.pdf")
