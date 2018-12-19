import numpy as np
import matplotlib.pyplot as plt

# définir matrices de rotation et de dilatation
rot = np.array([[0.94, -0.34], [0.34, 0.94]])
sca = np.array([[3.4, 0], [0, 2]])
# générer données classe 1
np.random.seed(150)
c1d = (np.random.randn(100,2)).dot(sca).dot(rot)

# générer données classe 2
c2d1 = np.random.randn(25,2)+[-10, 2]
c2d2 = np.random.randn(25,2)+[-7, -2]
c2d3 = np.random.randn(25,2)+[-2, -6]
c2d4 = np.random.randn(25,2)+[5, -7]

data = np.concatenate((c1d, c2d1, c2d2, c2d3, c2d4))

# générer étiquettes de classe
l1c = np.ones(100, dtype=int)
l2c = np.zeros(100, dtype=int)
labels = np.concatenate((l1c, l2c))

cmp = np.array(['r','g'])
#plt.scatter(data[:,0],data[:,1],c=cmp[labels],s=50,edgecolors='none')
#plt.show()

from sklearn.model_selection import train_test_split
score_train = []
score_test = []
score_pcm1_train = []
score_pcm1_test = []
score_pcm5_train = []
score_pcm5_test = []
for i in range(100):
    X_train1, X_test1, y_train1, y_test1 = train_test_split(data, labels, test_size=.33)
    #plt.scatter(X_train1[:,0],X_train1[:,1],c=cmp[y_train1],s=50,edgecolors='none')
    #plt.scatter(X_test1[:,0],X_test1[:,1],c='none',s=50,edgecolors=cmp[y_test1])
    #plt.show()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()

    # évaluation et affichage sur split1
    lda.fit(X_train1, y_train1)
    score_test.append(lda.score(X_test1, y_test1))
    score_train.append(lda.score(X_train1, y_train1))

    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(solver='lbfgs', alpha=1)
    # évaluation et affichage sur split1
    clf.fit(X_train1, y_train1)
    score_pcm1_train.append(clf.score(X_train1,y_train1))
    score_pcm1_test.append(clf.score(X_test1,y_test1))

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5)
    # évaluation et affichage sur split1
    clf.fit(X_train1, y_train1)
    score_pcm5_train.append(clf.score(X_train1, y_train1))
    score_pcm5_test.append(clf.score(X_test1, y_test1))

print(np.std(score_test))
print(np.std(score_train))
print(np.std(score_pcm5_train))
print(np.std(score_pcm5_test))
print(np.std(score_pcm1_train))
print(np.std(score_pcm1_test))


#plt.scatter(X_train1[:,0],X_train1[:,1],c=cmp[y_train1],s=50,edgecolors='none')
#plt.scatter(X_test1[:,0],X_test1[:,1],c='none',s=50,edgecolors=cmp[y_test1])
#nx, ny = 200, 100
#x_min, x_max = plt.xlim()
#y_min, y_max = plt.ylim()
#xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
#Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#Z = Z[:, 1].reshape(xx.shape)
#plt.contour(xx, yy, Z, [0.5])
#plt.show()


#print(clf.score(X_train1, y_train1))

#print(clf.score(X_test1, y_test1))

#plt.scatter(X_train1[:,0],X_train1[:,1],c=cmp[y_train1],s=50,edgecolors='none')
#plt.scatter(X_test1[:,0],X_test1[:,1],c='none',s=50,edgecolors=cmp[y_test1])
#nx, ny = 200, 200
#x_min, x_max = plt.xlim()
#y_min, y_max = plt.ylim()
#xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))
#Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
#Z = Z[:, 1].reshape(xx.shape)
#plt.contour(xx, yy, Z, [0.5])
#plt.show()


### second part

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
print(np.linalg.norm((reg.predict(X_train1.reshape(-1,1))-y_train1)))
print(np.linalg.norm((reg.predict(X_test1.reshape(-1,1))-y_test1)))
print(reg.score(X_test1.reshape(-1,1), y_test1))

plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
nx = 100
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
plt.plot(xx,reg.predict(xx.reshape(-1,1)),color='black')
plt.show()

coefdet = []
for et2 in range(1, 7):
    sca = np.array([[10, 0], [0, et2]])
    datar = rd.dot(sca).dot(rot)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(datar[:,0], datar[:,1], test_size=0.33)
    reg = linear_model.LinearRegression()
    reg.fit(X_train1.reshape(-1,1), y_train1)
    coefdet.append(reg.score(X_test1.reshape(-1,1), y_test1))

plt.plot(coefdet)
plt.show()

from sklearn.neural_network import MLPRegressor
clf = MLPRegressor(solver='lbfgs', alpha=1)

# évaluation et affichage sur split1
clf.fit(X_train1.reshape(-1,1), y_train1)
clf.score(X_train1.reshape(-1,1), y_train1)

clf.score(X_test1.reshape(-1,1), y_test1)

plt.scatter(X_train1,y_train1,s=50,edgecolors='none')
plt.scatter(X_test1,y_test1,c='none',s=50,edgecolors='blue')
x_min, x_max = plt.xlim()
xx = np.linspace(x_min, x_max, nx)
plt.plot(xx,clf.predict(xx.reshape(-1,1)),color='black')
plt.show()