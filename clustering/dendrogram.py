from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from utilities.utilities import plot_labelled_scatter
from scipy.cluster.hierarchy import ward, dendrogram

X, y = make_blobs(random_state=10, n_samples=10)
plot_labelled_scatter(X, y, ['Cluster 1', 'Cluster 2', 'Cluster 3'])
print(X)

plt.figure()
dendrogram(ward(X))
plt.show()
