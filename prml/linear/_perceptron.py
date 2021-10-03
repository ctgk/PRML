import numpy as np

from prml.linear._classifier import Classifier


class Perceptron(Classifier):
    """Perceptron model."""

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_epoch: int = 100,
    ):
        """Fit perceptron model on given input pair.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        y_train : np.ndarray
            training dependent variable (N,)
            binary -1 or 1
        max_epoch : int, optional
            maximum number of epoch (the default is 100)
        """
        self.w = np.zeros(np.size(x_train, 1))
        for _ in range(max_epoch):
            prediction = self.classify(x_train)
            error_indices = prediction != y_train
            x_error = x_train[error_indices]
            y_error = y_train[error_indices]
            idx = np.random.choice(len(x_error))
            self.w += x_error[idx] * y_error[idx]
            if ((x_train @ self.w) * y_train > 0).all():
                break

    def classify(self, x: np.ndarray):
        """Classify input data.

        Parameters
        ----------
        x : np.ndarray
            independent variable to be classified (N, D)

        Returns
        -------
        np.ndarray
            binary class (-1 or 1) for each input (N,)
        """
        return np.sign(x @ self.w).astype(np.int)
