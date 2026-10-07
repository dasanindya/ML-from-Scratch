import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

X,y =datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # EDA
# print(X_train.shape) # (120, 4) -> 120 training data, 4 features
# print(X_train[0])  # print sample
#
# print(y_train.shape) # y labels
# print(y_train) # print

# plt.figure()
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, edgecolor = 'k', s = 20)
# # 2 features - X_train[:, 0], X_train[:, 1]: These are the x and y coordinates for the points. It selects the first feature (column 0) and the second feature (column 1) of your training data array X_train.
# # c=y_train: This maps the color of each point to its corresponding class label or target value in y_train. Points with the same label will have the same color.
# # cmap=cmap: This applies a specific colormap (which you must define earlier in your code) to translate the y_train numbers into specific visual colors.
# # edgecolor='k': This adds a black outline ('k' stands for black in matplotlib) around each circle, making overlapping data points easier to distinguish.
# # s=20: This sets the size of the marker (the dots) to 20 points.
# plt.show()

from knn import KNN
clf = KNN(k=5) # usually odd number
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

acc = np.sum(prediction == y_test) / len(y_test)
print(acc)


# # Counter
# from collections import Counter
# a = [1,1,1,5,5,1,2,2]
# most_common = Counter(a).most_common(1)
# print(most_common)

