import pandas as pd
from utilities.utilities import plot_labelled_scatter
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS

# Our sample fruits dataset
fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X_fruits = fruits[['mass', 'width', 'height', 'color_score']]
y_fruits = fruits['fruit_label'] - 1

# each feature should be centered (zero mean) and with unit variance
scaler = StandardScaler()
X_fruits_normalized = scaler.fit_transform(X_fruits)

mds = MDS(n_components=2)

X_fruits_mds = mds.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_fruits_mds, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First MDS feature')
plt.ylabel('Second MDS feature')
plt.title('Fruit sample dataset MDS')
