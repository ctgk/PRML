import numpy as np

from prml.linear._classifier import Classifier


class LogisticRegression(Classifier):
    """Logistic regression model.

    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    @staticmethod
    def _sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        max_iter: int = 100,
    ):
        """Maximum likelihood estimation of logistic regression model.

        Parameters
        ----------
        x_train : (N, D) np.ndarray
            training data independent variable
        y_train : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of parameter update iteration (the default is 100)
        """
        w = np.zeros(np.size(x_train, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(x_train @ w)
            grad = x_train.T @ (y - y_train)
            hessian = (x_train.T * y * (1 - y)) @ x_train
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def proba(self, x: np.ndarray):
        """Return probability of input belonging class 1.

        Parameters
        ----------
        x : (N, D) np.ndarray
            Input independent variable

        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        return self._sigmoid(x @ self.w)

    def classify(self, x: np.ndarray, threshold: float = 0.5):
        """Classify input data.

        Parameters
        ----------
        x : (N, D) np.ndarray
            Input independent variable to be classified
        threshold : float, optional
            threshold of binary classification (default is 0.5)

        Returns
        -------
        (N,) np.ndarray
            binary class for each input
        """
        return (self.proba(x) > threshold).astype(int)
