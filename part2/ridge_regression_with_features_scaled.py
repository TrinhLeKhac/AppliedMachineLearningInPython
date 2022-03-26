import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utilities.utilities import load_crime_dataset
import warnings

warnings.filterwarnings('ignore')

scaler = MinMaxScaler()

from sklearn.linear_model import Ridge

X_crime, y_crime = load_crime_dataset()
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state=0, test_size=0.25)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linrigde = Ridge(alpha=20).fit(X_train_scaled, y_train)
print('Crime dataset')
print('Number of features: {}'.format(X_train.shape[0]))
print('Ridge regression linear model with coeff: {}'.format(linrigde.coef_))
print('Ridge regression linear model with intercept: {}'.format(linrigde.intercept_))
print('R-squared score(training): {:.2f}'.format(linrigde.score(X_train, y_train)))
print('R-squared score(testing): {:.2f}'.format(linrigde.score(X_test, y_test)))
print('Number of non-zero features: {}'.format(np.sum(linrigde.coef_ != 0)))

# Ridge regression effect by regularization(alpha)
print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha=this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\n'
          'num abs(coeff) > 1.0: {}, \
          r-squared training: {:.2f}, r-squared test: {:.2f}\n'
          .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
