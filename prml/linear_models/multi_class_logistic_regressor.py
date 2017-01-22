import numpy as np


class MultiClassLogisticRegressor(object):

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

    def _softmax(self, a):
        """
        softmax function

        Parameters
        ----------
        a : ndarray (..., n_classes)
            activations

        Returns
        -------
        output : ndarray (...,)
            output of softmax function
        """
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, X, t, iter_max=100):
        """
        perform gradient descent algorithm to estimate weight parameter

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        w : ndarray (n_features, n_classes)
            estimated paramters
        n_iter : int
            number iterations took until convergence
        """
        assert X.ndim == 2
        assert t.ndim == 1
        n_classes = np.max(t) + 1
        T = np.eye(n_classes)[t]
        self.w = np.zeros((np.size(X, 1), n_classes))
        I = np.eye(len(self.w))[:, :, None]
        for i in range(iter_max):
            w = np.copy(self.w)
            y = self.predict_proba(X)
            grad = X.T @ (y - T) + self.alpha * w
            hessian = np.einsum('ink,nj->ijk', X.T[:, :, None] * y * (1 - y), X) + self.alpha * I
            try:
                self.w -= np.linalg.solve(hessian.T, grad.T).T
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, self.w):
                break
        self.n_iter = i + 1

    def predict(self, X):
        """
        predict class label of each input datum

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        labels : ndarray (sample_size,)
            predicted labels
        """
        return np.argmax(self.predict_proba(X), axis=-1)

    def predict_proba(self, X):
        """
        computer probability for each class

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size, n_classes)
            probability for each class
        """
        return self._softmax(X @ self.w)
