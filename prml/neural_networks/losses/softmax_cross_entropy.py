import numpy as np
from .loss import Loss


class SoftmaxCrossEntropy(Loss):
    """
    sum of cross entropy for one-of-k coded data
    normalization of softmax activation can be taken along arbitrary axis

    softmax activation
    y_i = exp(x_i) / sum_j{exp(x_j)}
    cross_entropy_i = -t_i * log(y_i)

    Parameters
    ----------
    axis : int
        axis to normalize softmax activation along
    x : ndarray
        input logit
    t : ndarray
        corresponding target in one-of-k coding format
    """

    def __init__(self, axis=-1):
        """
        construct softmax cross entropy function

        Parameters
        ----------
        axis : int
            axis to normalize softmax activation along
        """
        self.axis = axis

    def __call__(self, x, t):
        """
        compute sum of cross entropies

        Parameters
        ----------
        x : ndarray
            input logit
        t : ndarray
            corresponding target in one-of-k coding format

        Returns
        -------
        output : float
            sum of cross entropies
        """
        y = self.forward(x)
        np.clip(y, 1e-10, 1, out=y)
        return np.sum(-t * np.log(y))

    def forward(self, x):
        """
        softmax function along the given axis
        exp(x_i) / sum_j{exp(x_j)}

        Parameters
        ----------
        x : ndarray
            input logit

        Returns
        -------
        y : ndarray
            softmax activation
        """
        y = np.exp(x - np.max(x, self.axis, keepdims=True))
        y /= np.sum(y, self.axis, keepdims=True)
        return y

    def backward(self, x, t):
        """
        compute input errors

        Parameters
        ----------
        x : ndarray
            input logit
        t : ndarray
            corresponding target in one-of-k coding format
        Returns
        -------
        delta : ndarray
            input errors
        """
        y = self.forward(x)
        return y - t
