import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X = fruits[['height', 'width', 'mass']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Score: {}'.format(knn.score(X_test, y_test)))

# Predict
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
print('Prediction: {}'.format(lookup_fruit_name[fruit_prediction[0]]))