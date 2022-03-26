import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_R1, y_R1 = make_regression(n_samples=1000, n_features=1, n_informative=1, bias=150, noise=30, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0, test_size=0.25)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print('Linear model coeff(w): {}'.format(linreg.coef_))
print('Linear model intercept(b): {}'.format(linreg.intercept_))
print('R-squared score(training): {}'.format(linreg.score(X_train, y_train)))
print('R-squared score(testing): {}'.format(linreg.score(X_test, y_test)))

plt.scatter(X_R1, y_R1, marker='o', s=10, alpha=0.5)
plt.plot(X_R1, linreg.coef_*X_R1+linreg.intercept_, 'r-')
plt.title('Least-squares linear regression')
plt.xlabel('Features value(x)')
plt.ylabel('Target values(y)')
plt.show()