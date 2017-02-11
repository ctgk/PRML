import numpy as np
from .loss import Loss


class SigmoidCrossEntropy(Loss):
    """
    sum of cross entropies for binary data

    logistic sigmoid
    y_i = 1 / (1 + exp(-x_i))
    cross_entropy_i = -t_i * log(y_i) - (1 - t_i) * log(1 - y_i)

    Parameters
    ----------
    x : ndarary
        input logit
    t : ndarray
        corresponding target binaries
    """

    def __call__(self, x, t):
        """
        cross entropy between target and sigmoid transformed input

        Parameters
        ----------
        x : ndarray
            input logit
        t : ndarray
            correponding target binaries

        Returns
        -------
        output : float
            sum of cross entropies
        """
        # y = self.forward(x)
        # np.clip(y, 1e-10, 1 - 1e-10, out=y)
        # return np.sum(-t * np.log(y) - (1 - t) * np.log(1 - y))
        return np.sum(np.maximum(x, 0) - t * x + np.log1p(np.exp(-np.abs(x))))

    def forward(self, x):
        """
        element-wise logistic sigmoid function

        Parameters
        ----------
        x : ndarray
            input logit

        Returns
        -------
        output : ndarray
            logistic sigmoid of each element
        """
        return np.tanh(x * 0.5) * 0.5 + 0.5

    def backward(self, x, t):
        """
        derivatives of the cost function with respect to the input

        Parameters
        ----------
        x : ndarray
            input logit
        t : ndarray
            corresponding target binaries

        Returns
        -------
        delta : ndarray
            input errors
        """
        y = self.forward(x)
        return y - t
