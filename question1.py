import numpy as np
import matplotlib.pyplot as plt

# Question :
# Générez d’autres découpages apprentissage / test (avec une même valeur test_size=0.33) et examinez la variabilité des résultats.
def getData():
    # définir matrices de rotation et de dilatation
    rot = np.array([[0.94, -0.34], [0.34, 0.94]])
    sca = np.array([[3.4, 0], [0, 2]])
    # générer données classe 1
    np.random.seed(np.random.randint(150))
    c1d = (np.random.randn(100, 2)).dot(sca).dot(rot)

    # générer données classe 2
    c2d1 = np.random.randn(25, 2) + [-10, 2]
    c2d2 = np.random.randn(25, 2) + [-7, -2]
    c2d3 = np.random.randn(25, 2) + [-2, -6]
    c2d4 = np.random.randn(25, 2) + [5, -7]

    data = np.concatenate((c1d, c2d1, c2d2, c2d3, c2d4))

    # générer étiquettes de classe
    l1c = np.ones(100, dtype=int)
    l2c = np.zeros(100, dtype=int)
    labels = np.concatenate((l1c, l2c))
    return data, labels

def testDecoupage(data, labels,decoupage = 0.33):
    # decoupage data : apprentissage / test
    from sklearn.model_selection import train_test_split

    X_train1, X_test1, y_train1, y_test1 = train_test_split(data, labels, test_size=decoupage)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()

    # évaluation et affichage sur split1
    lda.fit(X_train1, y_train1)
    print("decoupage : "+str(decoupage)+", score train : "+str(lda.score(X_train1, y_train1)))
    print("decoupage : "+str(decoupage)+", score test : "+str(lda.score(X_test1, y_test1)))
    # plot regression line
    cmp = np.array(['r', 'g'])
    plt.scatter(X_train1[:, 0], X_train1[:, 1], c=cmp[y_train1], s=50, edgecolors='none')
    plt.scatter(X_test1[:, 0], X_test1[:, 1], c='none', s=50, edgecolors=cmp[y_test1])
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])  ##TODO : check np.c_
    Z = Z[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, Z, [0.5])
    plt.show()

if __name__ =="__main__":
    for i in range(10):
        data, labels = getData()
        testDecoupage(data, labels, decoupage=0.33)
