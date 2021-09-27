import numpy as np

from prml.linear._classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer


class SoftmaxRegression(Classifier):
    """Softmax regression model.

    aka
    multinomial logistic regression,
    multiclass logistic regression,
    maximum entropy classifier.

    y = softmax(X @ W)
    t ~ Categorical(t|y)
    """

    @staticmethod
    def _softmax(a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_iter: int = 100,
        learning_rate: float = 0.1,
    ):
        """Maximum likelihood estimation of the parameter.

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) or (N, K) np.ndarray
            training dependent variable
            in class index or one-of-k encoding
        max_iter : int, optional
            maximum number of iteration (the default is 100)
        learning_rate : float, optional
            learning rate of gradient descent (the default is 0.1)
        """
        if y_train.ndim == 1:
            y_train = LabelTransformer().encode(y_train)
        self.n_classes = np.size(y_train, 1)
        w = np.zeros((np.size(x_train, 1), self.n_classes))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._softmax(x_train @ w)
            grad = x_train.T @ (y - y_train)
            w -= learning_rate * grad
            if np.allclose(w, w_prev):
                break
        self.w = w

    def proba(self, x: np.ndarray):
        """Return probability of input belonging each class.

        Parameters
        ----------
        x : np.ndarray
            Input independent variable (N, D)

        Returns
        -------
        np.ndarray
            probability of each class (N, K)
        """
        return self._softmax(x @ self.w)

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
        return np.argmax(self.proba(x), axis=-1)
