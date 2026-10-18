import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import random_sample

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K = 5, max_iters = 100, plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of indices for each cluster
        self.cluster = [[] for _ in range(self.K)]

        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace = False)

        # optimize clusters




