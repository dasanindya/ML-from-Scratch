import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state=4)
# n_samples=100: Generates 100 data points (rows).
# n_features=1: Creates a single independent variable
# noise=20: Adds Gaussian noise with a standard deviation of 20 to the output. N(0, sigma^2), here sigma = 20
# random_state=4: reproducibility

# fig = plt.figure()
# plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_train.shape)
# print(y_train.shape)

from linear_regression import LinearRegression

regressor = LinearRegression(lr=0.001, n_iters=1000)
regressor.fit(X_train, y_train)

predicted = regressor.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line = regressor.predict(X_train)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s = 10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s = 10)
plt.plot(X_train, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()