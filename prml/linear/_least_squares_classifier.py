import numpy as np

from prml.linear._classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer


class LeastSquaresClassifier(Classifier):
    """Least squares classifier model.

    X : (N, D)
    W : (D, K)
    y = argmax_k X @ W
    """

    def __init__(self, w: np.ndarray = None):
        """Initialize least squares classifier model.

        Parameters
        ----------
        w : np.ndarray, optional
            Initial parameter, by default None
        """
        self.w = w

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """Least squares fitting for classification.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        y_train : np.ndarray
            training dependent variable
            in class index (N,) or one-of-k coding (N,K)
        """
        if y_train.ndim == 1:
            y_train = LabelTransformer().encode(y_train)
        self.w = np.linalg.pinv(x_train) @ y_train

    def classify(self, x: np.ndarray):
        """Classify input data.

        Parameters
        ----------
        x : np.ndarray
            independent variable to be classified (N, D)

        Returns
        -------
        np.ndarray
            class index for each input (N,)
        """
        return np.argmax(x @ self.w, axis=-1)
