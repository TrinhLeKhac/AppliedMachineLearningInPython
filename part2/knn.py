import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
feature_fruits = ['height', 'width', 'mass', 'color_score']
X_fruits = fruits[feature_fruits]
y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0, test_size=0.25)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on testing set: {:.2f}'.format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.7]]
example_fruit_scale = scaler.transform(example_fruit)
print('Predict type of fruit for: {} is {}'.format(example_fruit, target_names_fruits[knn.predict(example_fruit_scale)[0]-1]))