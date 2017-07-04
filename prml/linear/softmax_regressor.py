import numpy as np
from prml.linear.classifier import Classifier
from prml.random.gaussian import Gaussian
from prml.random.random import RandomVariable


class SoftmaxRegressor(Classifier):
    """
    Softmax regression model
    aka multinomial logistic regression,
    multiclass logistic regression, or maximum entropy classifier.

    y = softmax(X @ W)
    t ~ Categorical(t|y)
    """

    def __init__(self, W=None):
        """
        construct softmax regression model

        Parameters
        ----------
        W : (n_features, n_classes) np.ndarray
            weight parameter of each feature for each class
        """
        self.W = W
        if isinstance(W, RandomVariable):
            self.W_prior = W

    def __setattr__(self, name, value):
        if name is "W":
            if isinstance(value, np.ndarray):
                assert value.ndim == 2
                self.n_features, self.n_classes = value.shape
                object.__setattr__(self, name, value)
            elif isinstance(value, Gaussian):
                assert value.mean is not None
                assert value.precision is not None
                assert value.ndim == 2
                self.n_features, self.n_classes = value.shape
                object.__setattr__(self, name, value)
            else:
                if value is not None:
                    raise ValueError(
                        "{} is not supported for W".format(type(value))
                    )
                object.__setattr__(self, name, None)
        elif name is "n_features":
            if not hasattr(self, "n_features"):
                object.__setattr__(self, name, value)
            else:
                assert self.n_features == value
        elif name is "n_classes":
            if not hasattr(self, "n_classes"):
                object.__setattr__(self, name, value)
            else:
                assert self.n_classes == value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        if hasattr(self, "W_prior"):
            reprval = (
                "Likelihood Categorical(t|softmax(X@w))\n"
                "Prior W~{}"
                .format(self.W_prior)
            )
            if isinstance(self.W, np.ndarray):
                return "MAP estimate\n{}\n".format(self.W) + reprval
            else:
                if isinstance(self.W, RandomVariable):
                    if self.W_prior == self.W:
                        return reprval
                    return "Posterior W~{}\n".format(self.W) + reprval
        else:
            return "Categorical(t|softmax(X@{}))".format(self.W)

    def _ml(self, X, t, max_iter=100, learning_rate=0.1):
        self.n_classes = np.max(t) + 1
        T = np.eye(self.n_classes)[t]
        W = np.zeros((np.size(X, 1), self.n_classes))
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - T)
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W

    def _map(self, X, t, max_iter=100, learning_rate=0.1):
        assert isinstance(self.W, Gaussian)
        self.n_classes = np.max(t) + 1
        T = np.eye(self.n_classes)[t]
        W = np.zeros(self.W.shape)
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - T) + self.W.precision * W
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W

    def _softmax(self, a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def _proba(self, X):
        if isinstance(self.W, np.ndarray):
            y = self._softmax(X @ self.W)
        elif isinstance(self.W, Gaussian):
            y = self._softmax(X @ self.W.mean)
        return y

    def _classify(self, X):
        proba = self._proba(X)
        label = np.argmax(proba, axis=-1)
        return label
