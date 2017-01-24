import numpy as np
from prml.distributions import GaussianDistribution


class LinearDiscriminantAnalyzer(object):

    def fit(self, X, t):
        """
        estimate decision boundary by Linear Discriminant Analysis

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels 0 or 1

        Attributes
        ----------
        w : ndarray (n_features,)
            normal vector of hyperplane defining decision boundary
        threshold : float
            boundary value in projected space
        """
        assert X.ndim == 2
        assert t.ndim == 1
        assert np.max(t) == 1
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        cov_inclass = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)
        self.w = np.linalg.inv(cov_inclass) @ (m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=1e-10)
        g0 = GaussianDistribution()
        g0.fit(X0 @ self.w)
        g1 = GaussianDistribution()
        g1.fit(X1 @ self.w)
        a = g1.var - g0.var
        b = g0.var * g1.mean - g1.var * g0.mean
        c = (
            g1.var * g0.mean ** 2 - g0.var * g1.mean ** 2
            - g1.var * g0.var * np.log(g1.var / g0.var))
        self.threshold = (np.sqrt(b ** 2 - a * c) - b) / a

    def predict(self, X):
        """
        predict class labels

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        labels : ndarray (sample_size,)
            class labels
        """
        assert X.ndim == 2
        return (X @ self.w > self.threshold).astype(np.int)
