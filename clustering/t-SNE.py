import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from utilities.utilities import plot_labelled_scatter
import warnings
warnings.filterwarnings('ignore')

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X_fruits = fruits[['mass', 'width', 'height', 'color_score']].values
y_fruits = fruits['fruit_label'] - 1

scaler = MinMaxScaler()
X_fruits_normalized = scaler.fit_transform(X_fruits)

tsne = TSNE(random_state=0)

X_tsne = tsne.fit_transform(X_fruits_normalized)

plot_labelled_scatter(X_tsne, y_fruits, ['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second t-SNE feature')
plt.title('Fruits dataset t-SNE');
