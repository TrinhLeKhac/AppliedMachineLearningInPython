import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y=True)

# Before applying PCA, each feature should be centered (zero mean) and with unit variance
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_cancer)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

from utilities.utilities import plot_labelled_scatter

plot_labelled_scatter(X_pca, y_cancer, ['malignant', 'benign'])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Breast Cancer Dataset PCA (n_components = 2)')

fig = plt.figure(figsize=(8, 4))
plt.imshow(pca.components_, interpolation='none', cmap='plasma')
feature_names = list(cancer.feature_names)
print(pca.components_)

plt.gca().set_xticks(np.arange(0.5, len(feature_names)))
plt.gca().set_yticks(np.arange(0.5, 2))
plt.gca().set_xticklabels(feature_names, rotation=90, ha='left', fontsize=12)
plt.gca().set_yticklabels(['First PC', 'Second PC'], va='bottom', fontsize=12)

plt.colorbar(orientation='horizontal', ticks=[pca.components_.min(), 0,
                                              pca.components_.max()], pad=0.65)
