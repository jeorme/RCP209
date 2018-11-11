import numpy as np
import matplotlib.pyplot as plt
# définir matrices de rotation et de dilatation
rot = np.array([[0.94, 0.34], [-0.34, 0.94]])
sca = np.array([[10, 0], [0, 1]])
# générer données classe 1
np.random.seed(60)
rd = np.random.randn(60,2)
datar = rd.dot(sca).dot(rot)
plt.scatter(datar[:,0],datar[:,1],s=50,edgecolors='none')
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
#plt.show()

from sklearn import linear_model
reg = linear_model.LinearRegression()

# évaluation et affichage sur split1
reg.fit(X_train1.reshape(-1,1), y_train1)
# attention, pas erreur mais coeff. détermination !
print(reg.score(X_train1.reshape(-1,1), y_train1))

print(reg.score(X_test1.reshape(-1,1), y_test1))

plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
nx = 100
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
plt.plot(xx,reg.predict(xx.reshape(-1,1)),color='black')
#plt.show()
y_predict_train = reg.predict(X_train1.reshape(-1,1))
y_predict_test = reg.predict(X_test1.reshape(-1,1))
print(np.dot(y_predict_train,y_train1)/(len(y_predict_train)))
print(mean_squared_error(y_predict_train,y_train1))
print(mean_squared_error(y_predict_test,y_test1))

from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='lbfgs', alpha=1e-5)

# évaluation et affichage sur split1
clf.fit(X_train1.reshape(-1,1), y_train1)
print(clf.score(X_train1.reshape(-1,1), y_train1))

print(clf.score(X_test1.reshape(-1,1), y_test1))

plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
plt.plot(xx,clf.predict(xx.reshape(-1,1)),color='black')
plt.show()