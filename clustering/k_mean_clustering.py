import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from utilities.utilities import plot_labelled_scatter
from sklearn.preprocessing import MinMaxScaler

# data generate
X, y = make_blobs(random_state=10)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

plot_labelled_scatter(X, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

# data fruit
fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X_fruits = fruits[['mass', 'width', 'height', 'color_score']].values
y_fruits = fruits['fruit_label'] - 1

scaler = MinMaxScaler()
X_fruits_normalized = scaler.fit_transform(X_fruits)

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_fruits_normalized)

plot_labelled_scatter(X_fruits_normalized, kmeans.labels_,
                      ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
