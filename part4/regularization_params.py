from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from utilities.utilities import plot_class_regions_for_classifier_subplot
import warnings

warnings.filterwarnings('ignore')

X_D2, y_D2 = make_blobs(n_samples=100, n_features=2,
                        centers=8, cluster_std=1.3,
                        random_state=4)
y_D2 = y_D2 % 2
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))

for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation='tanh',
                          alpha=this_alpha,
                          hidden_layer_sizes=[100, 100],
                          random_state=0).fit(X_train, y_train)

    title = 'Dataset 2: NN classifier, alpha = {:.3f} '.format(this_alpha)
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train,
                                              X_test, y_test, title, axis)
    plt.tight_layout()
plt.show()