import numpy as np


class LogisticRegressor(object):

    def __init__(self, alpha=1.):
        """
        construct logistic regressor

        Parameters
        ----------
        alpha : float
            precision parameter for prior distribution, which denotes regularization strength
        """
        assert isinstance(alpha, float) or isinstance(alpha, int)
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t, iter_max=100):
        """
        Iterative reweighted least squares method to estimate parameter

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels
        iter_max : int
            number of maximum iterations

        Attributes
        ----------
        w : ndarray (n_features,)
            estimated parameter
        n_iter : int
            number of iterations took until convergence
        """
        assert X.ndim == 2
        assert t.ndim == 1
        self.w = np.zeros(np.size(X, 1))
        I = np.eye(len(self.w))
        for i in range(iter_max):
            w = np.copy(self.w)
            y = self.predict_proba(X)
            grad = X.T @ (y - t) + self.alpha * w
            hessian = (X.T * y * (1 - y)) @ X + self.alpha * I
            try:
                self.w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, self.w):
                break
        self.n_iter = i + 1

    def predict(self, X, threshold=0.5):
        """
        predict binary class label for each input

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input
        threshold : float
            threshold value to be predicted as positive

        Returns
        -------
        output : ndarray (sample_size,)
            binary class labels
        """
        assert X.ndim == 2
        assert isinstance(threshold, float)
        return (self.predict_proba(X) > threshold).astype(np.int)

    def predict_proba(self, X):
        """
        probability of input belonging class one

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        output : ndarray (sample_size,)
            probability of class one for each input
        """
        assert X.ndim == 2
        return self._sigmoid(X @ self.w)
