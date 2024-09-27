import numpy as np


class RelevanceVectorClassifier(object):

    def __init__(self, kernel, alpha=1.):
        """
        construct relevance vector classifier

        Parameters
        ----------
        kernel : Kernel
            kernel function to compute components of feature vectors
        alpha : float
            initial precision of prior weight distribution
        """
        self.kernel = kernel
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _map_estimate(self, x, t, w, n_iter=10):
        for _ in range(n_iter):
            y = self._sigmoid(x @ w)
            g = x.T @ (y - t) + self.alpha * w
            H = (x.T * y * (1 - y)) @ x + np.diag(self.alpha)
            w -= np.linalg.solve(H, g)
        return w, np.linalg.inv(H)

    def fit(self, x, t, iter_max=100):
        """
        maximize evidence with respect ot hyperparameter

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input
        t : (sample_size,) ndarray
            corresponding target
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        x : (N, n_features) ndarray
            relevance vector
        t : (N,) ndarray
            corresponding target
        alpha : (N,) ndarray
            hyperparameter for each weight or training sample
        cov : (N, N) ndarray
            covariance matrix of weight
        mean : (N,) ndarray
            mean of each weight
        """
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2
        assert t.ndim == 1
        Phi = self.kernel(x, x)
        N = len(t)
        self.alpha = np.zeros(N) + self.alpha
        mean = np.zeros(N)
        for _ in range(iter_max):
            param = np.copy(self.alpha)
            mean, cov = self._map_estimate(Phi, t, mean, 10)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            if np.allclose(param, self.alpha):
                break
        mask = self.alpha < 1e8
        self.x = x[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.x, self.x)
        mean = mean[mask]
        self.mean, self.covariance = self._map_estimate(Phi, self.t, mean, 100)

    def predict(self, x):
        """
        predict class label

        Parameters
        ----------
        x : (sample_size, n_features)
            input

        Returns
        -------
        label : (sample_size,) ndarray
            predicted label
        """
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2
        phi = self.kernel(x, self.x)
        label = (phi @ self.mean > 0).astype(int)
        return label

    def predict_proba(self, x):
        """
        probability of input belonging class one

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        proba : (sample_size,) ndarray
            probability of predictive distribution p(C1|x)
        """
        if x.ndim == 1:
            x = x[:, None]
        assert x.ndim == 2
        phi = self.kernel(x, self.x)
        mu_a = phi @ self.mean
        var_a = np.sum(phi @ self.covariance * phi, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
