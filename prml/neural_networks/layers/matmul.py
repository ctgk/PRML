import numpy as np
import scipy.stats as st
from .layer import Layer


class MatMul(Layer):
    """Matrix multiplication"""

    def __init__(self, dim_in, dim_out, std=1., istrainable=True):
        """
        initialize this layer

        Parameters
        ----------
        dim_in : int
            dimensionality of input
        dim_out : int
            dimensionality of output
        std : float
            standard deviation of truncnorm distribution for initializing parameter
        istrainable : bool
            flag indicating trainable or not

        Attributes
        ----------
        param : ndarray (dim_in, dim_out)
            coefficient to be matrix multiplied to the input
        deriv : ndarray (dim_in, dim_out)
            derivative of a cost function with respect to the paramter
        """
        self.param = st.truncnorm(
            a=-2 * std, b=2 * std, scale=std).rvs((dim_in, dim_out))
        self.deriv = np.zeros_like(self.param)
        self.istrainable = istrainable

    def forward(self, x, training=False):
        """
        forward propagation
        x @ w

        Parameters
        ----------
        x : ndarray (sample_size, dim_in)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim_out)
            x @ w
        """
        if training:
            self.input = x
        return x @ self.param

    def backward(self, delta):
        """
        backpropagation of errors

        Parameters
        ----------
        delta : ndarray (sample_size, dim_out)
            output error

        Returns
        -------
        delta_in : ndarray (sample_size, dim_in)
            input error
        """
        delta_in = delta @ self.param.T
        if self.istrainable:
            self.deriv = self.input.T @ delta
        return delta_in
