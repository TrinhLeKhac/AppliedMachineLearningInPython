from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

X_F1, y_F1 = make_friedman1(n_samples=100, n_features=7, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_F1, y_F1, random_state=0)
lin_reg = LinearRegression().fit(X_train, y_train)

print('linear model coeff (w): {}'
      .format(lin_reg.coef_))
print('linear model intercept (b): {:.3f}'
      .format(lin_reg.intercept_))
print('R-squared score (training): {:.3f}'
      .format(lin_reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
      .format(lin_reg.score(X_test, y_test)))

print('\nNow we transform the original input data to add\n\
polynomial features up to degree 2 (quadratic)\n')
poly = PolynomialFeatures(degree=2)
X_F1_poly = poly.fit_transform(X_F1)

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state=0)
poly_reg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) linear model coeff (w):\n{}'
      .format(poly_reg.coef_))
print('(poly deg 2) linear model intercept (b): {:.3f}'
      .format(poly_reg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
      .format(poly_reg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
      .format(poly_reg.score(X_test, y_test)))

print('\nAddition of many polynomial features often leads to\n\
overfiting, so we often use polynomial features in combination\n\
with regression that has a regularization penalty, like ridge\n\
regression.\n')

X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1, random_state=0)
poly_ridge_reg = Ridge().fit(X_train, y_train)

print('(poly deg 2 + ridge) linear model coeff (w):\n{}'
      .format(poly_ridge_reg.coef_))
print('(poly deg 2 + ridge) linear model intercept (b): {:.3f}'
      .format(poly_ridge_reg.intercept_))
print('(poly deg 2 + ridge) R-squared score (training): {:.3f}'
      .format(poly_ridge_reg.score(X_train, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {:.3f}'
      .format(poly_ridge_reg.score(X_test, y_test)))
