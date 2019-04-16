import numpy as np
from prml.linear.classifier import Classifier


class Perceptron(Classifier):
    """
    Perceptron model
    """

    def fit(self, X, t, max_epoch=100):
        """
        fit perceptron model on given input pair

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,)
            training dependent variable
            binary -1 or 1
        max_epoch : int, optional
            maximum number of epoch (the default is 100)
        """
        self.w = np.zeros(np.size(X, 1))
        for _ in range(max_epoch):
            X_error=X[np.sign(X@self.w)!=t]
            t_error=t[np.sign(X@self.w)!=t]
            idx=np.random.choice(len(X_error))
            self.w+=X_error[idx]*t_error[idx]
            if (X@self.w*t>0).all():
                break

    def classify(self, X):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified

        Returns
        -------
        (N,) np.ndarray
            binary class (-1 or 1) for each input
        """
        return np.sign(X @ self.w).astype(np.int)
