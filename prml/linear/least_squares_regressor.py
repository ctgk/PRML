import numpy as np
from prml.linear.regressor import Regressor
from prml.random.multivariate_gaussian import MultivariateGaussian
from prml.random.random import RandomVariable


class LeastSquaresRegressor(Regressor):
    """
    Least squares regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def __init__(self, w=None, var=None, precision=None):
        """
        construct linear regression model

        Parameters
        ----------
        w : (n_features,) np.ndarray
            weight parameter of each feature
        var : int or float
            variance of noise
        precision : int or float
            precision of noise
        """
        self.w = w
        if isinstance(w, RandomVariable):
            self.w_prior = w
        if var is not None:
            self.var = var
        elif precision is not None:
            self.var = 1 / precision
        else:
            self.var = None

    @property
    def precision(self):
        return 1 / self.var

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
                assert value is None, (
                    "w must be either np.ndarray, Gaussian, or None"
                )
                object.__setattr__(self, name, value)
        elif name is "var":
            if isinstance(value, (int, float)):
                object.__setattr__(self, name, float(value))
            else:
                assert value is None, (
                    "var must be either int, float, or None"
                )
                object.__setattr__(self, name, value)
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
                "Likelihood Gaussian(t|mean=X@w, var={0.var})\n"
                "Prior w~{0.w_prior}"
                .format(self)
            )
            if isinstance(self.w, np.ndarray):
                return "MAP estimate {}\n".format(self.w) + string
            elif isinstance(self.w, RandomVariable):
                if self.w_prior == self.w:
                    return string
                return "Posterior w~{}\n".format(self.w) + string
        else:
            return "Gaussian(t|mean=X@{0.w}, var={0.var})".format(self)

    def _ml(self, X, t):
        self.w = np.linalg.pinv(X) @ t
        self.var = np.mean(np.square(X @ self.w - t))

    def _map(self, X, t):
        assert isinstance(self.w, MultivariateGaussian)
        w_var = np.linalg.inv(
            self.w.precision + X.T @ X / self.var
        )
        self.w = w_var @ (self.w.precision @ self.w.mean + X.T @ t / self.var)

    def _bayes(self, X, t):
        assert isinstance(self.w, MultivariateGaussian)
        w_var = np.linalg.inv(
            self.w.precision + X.T @ X / self.var
        )
        w_mean = w_var @ (self.w.precision @ self.w.mean + X.T @ t / self.var)
        self.w = MultivariateGaussian(mean=w_mean, var=w_var)

    def _empirical_bayes(self, X, t, max_iter=100):
        assert isinstance(self.w, MultivariateGaussian)
        assert (self.w.mean == 0).all()
        assert isinstance(self.w_prior.precision_, (int, float))

        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        for _ in range(max_iter):
            params = [self.w_prior.var_, self.var]

            w_var = np.linalg.inv(
                self.w_prior.precision + X.T @ X / self.var
            )
            w = self.precision * w_var @ X.T @ t

            gamma = np.sum(
                self.precision * eigenvalues
                / (self.precision * eigenvalues + self.w_prior.precision_)
            )
            self.w_prior.precision_ = float(gamma / (w @ w).clip(min=1e-10))
            self.var = float(np.sum(np.square(t - X @ w)) / (len(t) - gamma))
            if np.allclose(params, [self.w_prior.precision_, self.var]):
                break
        self.w = MultivariateGaussian(mean=w, var=w_var)

    def _hierarchical_bayes(self, X, t):
        pass

    def log_evidence(self, X, t):
        """
        log evidence function
        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data

        Returns
        -------
        output : float
            log evidence
        """
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.w_prior.precision_)
            + len(t) * np.log(self.precision)
            - self.precision * np.sum(np.square(t - X @ self.w.mean))
            - self.w_prior.precision_ * self.w.mean @ self.w.mean
            - np.linalg.slogdet(
                self.w_prior.precision_ * np.eye(len(M))
                + self.precision * M)[1]
            - len(t) * np.log(2 * np.pi)
        )

    def _predict(self, X, return_std=False, sample_size=None):
        if isinstance(self.w, np.ndarray):
            y = X @ self.w
            if return_std:
                y_std = np.sqrt(self.var) + np.zeros_like(y)
                return y, y_std
            return y
        elif isinstance(self.w, MultivariateGaussian):
            if isinstance(sample_size, int):
                w_sample = self.w.draw(sample_size)
                y = X @ w_sample.T
                return y
            y = X @ self.w.mean
            if return_std:
                y_var = self.var + np.sum(X @ self.w.var * X, axis=1)
                y_std = np.sqrt(y_var)
                return y, y_std
            return y
        else:
            raise AttributeError
