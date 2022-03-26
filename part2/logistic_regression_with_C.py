import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utilities.utilities import plot_class_regions_for_classifier_subplot

fruits = pd.read_table('../data/fruit_data_with_colors.txt')
X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']
y_fruits_apple = y_fruits_2d == 1

X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d.values, y_fruits_apple.values, random_state=0)
print(X_train.shape, y_train.shape)

fig, subaxes = plt.subplots(3, 1, figsize=(4, 10))

for this_C, subplot in zip([0.1, 1, 100], subaxes):
    clf = LogisticRegression(C=this_C).fit(X_train, y_train)
    title = 'Logistic regression (apple vs rest), C = {:.3f}'.format(this_C)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subplot)
plt.tight_layout()
plt.show()
