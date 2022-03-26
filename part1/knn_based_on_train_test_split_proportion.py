import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X = fruits[['height', 'width', 'mass']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

proportions = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn = KNeighborsClassifier(n_neighbors=5)
plt.figure()

for p in proportions:
    scores = []
    for i in range(1, 100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-p)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(p, np.mean(scores), 'bo')

plt.xlabel('Training set proportion(%)')
plt.ylabel('Accuracy')
plt.show()