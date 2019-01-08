import numpy
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from kerasHelper import get_minst
# getData
from kerasHelper import convexHulls, best_ellipses, neighboring_hit, visualization

test = [0.1895129785188016, 0.18505048690932335, 0.1439318164498207, 0.14834934621000942, 0.13853697171598417, 0.1402252191805333, 0.1778971211656891, 0.15112501905197398, 0.1349064515238952, 0.12704903978052134]
print(numpy.sum(test))

X_train, y_train, X_test, y_test =get_minst()
X_train_embedded = TSNE(init="pca",perplexity=30,verbose=2).fit_transform(X_train[:1000])
# Function Call
convex_hulls= convexHulls(X_train_embedded, y_train[:1000])
ellipses = best_ellipses(X_train_embedded, y_train[:1000])
nh = neighboring_hit(X_train_embedded, y_train[:1000])

visualization(X_train_embedded, y_train[:1000], convex_hulls, ellipses ,"EXOCNN", nh)

X_train_PCA = PCA(n_components=2).fit_transform(X_train[:1000])
convex_hulls= convexHulls(X_train_PCA, y_train[:1000])
ellipses = best_ellipses(X_train_PCA, y_train[:1000])
nh = neighboring_hit(X_train_PCA, y_train[:1000])

visualization(X_train_PCA, y_train[:1000], convex_hulls, ellipses ,"EXOPCA", nh)