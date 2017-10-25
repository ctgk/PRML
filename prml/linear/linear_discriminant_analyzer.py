import numpy as np
from prml.linear.classifier import Classifier
from prml.rv.gaussian import Gaussian


class LinearDiscriminantAnalyzer(Classifier):
    """
    Linear discriminant analysis model
    """

    def _fit(self, X, t, clip_min_norm=1e-10):
        self._check_input(X)
        self._check_target(t)
        self._check_binary(t)
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        cov_inclass = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)
        self.w = np.linalg.solve(cov_inclass, m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=clip_min_norm)
        g0 = Gaussian()
        g0.fit((X0 @ self.w)[:, None])
        g1 = Gaussian()
        g1.fit((X1 @ self.w)[:, None])
        a = g1.var - g0.var
        b = g0.var * g1.mu - g1.var * g0.mu
        c = (
            g1.var * g0.mu ** 2 - g0.var * g1.mu ** 2
            - g1.var * g0.var * np.log(g1.var / g0.var)
        )
        self.threshold = (np.sqrt(b ** 2 - a * c) - b) / a

    def transform(self, X):
        """
        project data

        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            input data

        Returns
        -------
        y : (sample_size, 1) np.ndarray
            projected data
        """
        if not hasattr(self, "w"):
            raise AttributeError("perform fit method to estimate linear projection")
        return X @ self.w

    def _classify(self, X):
        return (X @ self.w > self.threshold).astype(np.int)
