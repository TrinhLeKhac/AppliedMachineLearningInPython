import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X = fruits[['height', 'width', 'mass']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

from utilities.utilities import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'uniform')
