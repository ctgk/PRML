import numpy as np


class LeastSquaresClassifier(object):

    def fit(self, X, t):
        """
        perform least squares algorithm for classification

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels

        Returns
        -------
        W : ndarray (n_features, n_classes)
            parameter estimated by least squares alg.
        """
        assert X.ndim == 2
        assert t.ndim == 1
        T = np.eye(int(np.max(t)) + 1)[t]
        self.W = np.linalg.pinv(X) @ T

    def predict(self, X):
        """
        predict corresponding label

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size)
            class labels
        """
        assert X.ndim == 2
        return np.argmax(X @ self.W, axis=1)
