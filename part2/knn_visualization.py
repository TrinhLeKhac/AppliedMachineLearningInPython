import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from utilities.utilities import plot_two_class_knn

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF', '#000000'])

from sklearn.datasets import make_regression

plt.figure()
plt.title('Sample regression problem with one input features')
X_R1, y_R1 = make_regression(n_samples=1000, n_features=1, n_informative=1, bias=150, noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker='o', s=100)
plt.show()

plt.figure()
plt.title('Sample binary classification problem with two variables features')
X_C2, y_C2 = make_classification(n_samples=1000,
                                 n_features=2,
                                 n_redundant=0,
                                 n_informative=2,
                                 n_clusters_per_class=1,
                                 flip_y=0.1,
                                 class_sep=0.5,
                                 random_state=0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2, marker='o', s=100, cmap=cmap_bold)
plt.show()

# more difficult synthetic dataset for classification
# with classes that are not linearly separable
X_D2, y_D2 = make_blobs(n_samples=1000, n_features=2, centers=8, cluster_std=1.3, random_state=0)
plt.scatter(X_D2[:, 0], X_D2[:, 1], c=y_D2, marker='o', s=100, cmap=cmap_bold)
plt.show()

X_cancer, y_cancer = load_breast_cancer(return_X_y=True)



