import numpy as np


class SquaredError(object):
    """
    cost function of sum of squared errors
    """

    def __call__(self, x, t):
        """
        sum of squared errors

        Parameters
        ----------
        x : ndarray
            input
        t : ndarray
            corresponding target

        Returns
        -------
        error : float
            sum of squared errors
        """
        return 0.5 * np.sum(np.square(x - t))

    def forward(self, x):
        """
        identity function

        Parameters
        ----------
        x : ndarray
            input

        Returns
        -------
        output : ndarray
            identity of input
        """
        return x

    def backward(self, x, t):
        """
        compute input errors

        Parameters
        ----------
        x : ndarray
            input
        t : ndarray
            corresponding target

        Returns
        -------
        delta : ndarray
            input errors
        """
        return x - t
