import numpy as np
from prml.linear.classifier import Classifier
from prml.random.random import RandomVariable
from prml.random.multivariate_gaussian import MultivariateGaussian


class LogisticRegressor(Classifier):
    """
    Logistic regression model
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    def __init__(self, w=None):
        """
        construct logistic regression model

        Parameters
        ----------
        w : (n_features,) np.ndarray or Gaussian
            weight parameter of each feature
        """
        self.w = w
        if isinstance(w, RandomVariable):
            self.w_prior = w

    def __setattr__(self, name, value):
        if name is "w":
            if isinstance(value, np.ndarray):
                assert value.ndim == 1
                self.n_features = value.size
                object.__setattr__(self, name, value)
            elif isinstance(value, MultivariateGaussian):
                assert value.mean is not None
                assert value.precision is not None
                if hasattr(value, "size"):
                    self.n_features = value.size
                object.__setattr__(self, name, value)
            else:
                if value is not None:
                    raise ValueError(
                        "{} is not supported for w".format(type(value))
                    )
                object.__setattr__(self, name, None)
        elif name is "n_features":
            if not hasattr(self, "n_features"):
                object.__setattr__(self, name, value)
            else:
                assert self.n_features == value
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        if hasattr(self, "w_prior"):
            string = (
                "Likelihood Bernoulli(t|sigmoid(X@w))\n"
                "Prior w~{0}"
                .format(self.w_prior)
            )
            if isinstance(self.w, np.ndarray):
                return "MAP estimate {}\n".format(self.w) + string
            else:
                if isinstance(self.w, RandomVariable):
                    if self.w_prior == self.w:
                        return string
                    return "Posterior w~{}\n".format(self.w) + string
        else:
            return "Bernoulli(t|sigmoid(X@{}))".format(self.w)

    def _ml(self, X, t, max_iter=100):
        self._check_binary(t)
        w = np.zeros(np.size(X, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t)
            hessian = (X.T * y * (1 - y)) @ X
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def _map(self, X, t, max_iter=100):
        self._check_binary(t)
        assert isinstance(self.w, MultivariateGaussian)
        w = np.zeros(np.size(X, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t) + self.w.precision @ w
            hessian = (X.T * y * (1 - y)) @ X + self.w.precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

    def _bayes(self, X, t, max_iter=100):
        self._check_binary(t)
        assert isinstance(self.w, MultivariateGaussian)
        w = np.zeros(np.size(X, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t) + self.w.precision @ w
            hessian = (X.T * y * (1 - y)) @ X + self.w.precision
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = MultivariateGaussian(mean=w, precision=hessian)

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def _proba(self, X, sample_size=None):
        if isinstance(self.w, np.ndarray):
            y = self._sigmoid(X @ self.w)
            return y
        elif isinstance(self.w, MultivariateGaussian):
            mu_a = X @ self.w.mean
            var_a = np.sum(X @ self.w.var * X, axis=1)
            y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
            return y
        elif isinstance(self.w, RandomVariable):
            w_sample = self.w.draw(sample_size)
            y = self._sigmoid(X @ w_sample.T)
            y = np.mean(y, axis=1)
            return y
        else:
            raise AttributeError

    def _classify(self, X, threshold=0.5, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = self.w.draw(sample_size)
            y = self._sigmoid(X @ w_sample.T)
            label = (y > threshold).astype(np.int)
            return label
        proba = self._proba(X)
        label = (proba > threshold).astype(np.int)
        return label
