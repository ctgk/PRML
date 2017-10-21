import numpy as np


class Classifier(object):
    """
    Base class for classifiers
    """

    def fit(self, X, t, **kwargs):
        """
        estimate parameters given the training dataset

        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            training data input
        t : (sample_size,) np.ndarray
            training data target
        """
        self._check_input(X)
        self._check_target(t)
        if hasattr(self, "_fit"):
            self._fit(X, t, **kwargs)
        else:
            raise NotImplementedError

    def classify(self, X, **kwargs):
        """
        classify inputs

        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            samples to classify

        Returns
        -------
        label : (sample_size,) np.ndarray
            label index for each sample
        """
        self._check_input(X)
        if hasattr(self, "_classify"):
            return self._classify(X, **kwargs)
        else:
            raise NotImplementedError

    def proba(self, X, **kwargs):
        """
        compute probability of input belonging each class

        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            samples to compute their probability

        Returns
        -------
        proba : (sample_size, n_classes) np.ndarray
            probability for each class
        """
        self._check_input(X)
        if hasattr(self, "_proba"):
            return self._proba(X, **kwargs)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) must be np.ndarray")
        if X.ndim != 2:
            raise ValueError("X(input) must be two dimensional array")
        if hasattr(self, "n_features") and self.n_features != np.size(X, 1):
            raise ValueError(
                "mismatch in dimension 1 of X(input) (size {} is different from {})"
                .format(np.size(X, 1), self.n_features)
            )

    def _check_target(self, t):
        if not isinstance(t, np.ndarray):
            raise ValueError("t(target) must be np.ndarray")
        if t.ndim != 1:
            raise ValueError("t(target) must be one dimensional array")
        if t.dtype != np.int:
            raise ValueError("dtype of t(target) must be np.int")
        if (t < 0).any():
            raise ValueError("t(target) must only has positive values")

    def _check_binary(self, t):
        if np.max(t) > 1 and np.min(t) >= 0:
            raise ValueError("t(target) must only has 0 or 1")

    def _check_binary_negative(self, t):
        n_zeros = np.count_nonzero(t == 0)
        n_ones = np.count_nonzero(t == 1)
        if n_zeros + n_ones != t.size:
            raise ValueError("t(target) must only has -1 or 1")
