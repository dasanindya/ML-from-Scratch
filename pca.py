from typing import Any

import numpy as np
from numpy import dtype, ndarray, signedinteger


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean # (N, D)

        # covariance, function needs sample as columns
        cov = np.cov(X.T) # (D, D)

        # eigen values and eigen vectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # eigenvalues: (D,)
        # eigenvectors: (D, D) -> columns represent the eigenvectors.

        # eigenvector v = [:, i] column vector, transpose for easier calculations
        eigenvectors = eigenvectors.T # (D, D) -> rows become the eigenvectors
        idxs = np.argsort(eigenvalues)[::-1] # descending order.
        # eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0: self.components] # (K, D)


    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T) # (M, D) x (D, K) -> (M, K)