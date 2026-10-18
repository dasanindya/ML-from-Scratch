from sklearn.datasets import make_blobs
from kmeans import KMeans
import numpy as np

X, y = make_blobs(
    centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150)
y_pred = k.predict(X)

from collections import Counter
print(Counter(y_pred))

