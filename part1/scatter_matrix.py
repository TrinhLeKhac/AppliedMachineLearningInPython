import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from sklearn.model_selection import train_test_split

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
cmap = cm.get_cmap("gnuplot")

scatter = pd.plotting.scatter_matrix(X_train,
                                     c=y_train,
                                     marker='o',
                                     s=50,
                                     hist_kwds={'bins': 15},
                                     figsize=(9, 9),
                                     cmap=cmap)
plt.show()
