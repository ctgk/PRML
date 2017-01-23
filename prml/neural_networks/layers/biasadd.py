import numpy as np
from .layer import Layer


class BiasAdd(Layer):
    """Add bias"""

    def __init__(self, dim, value=0., istrainable=True):
        """
        initialize parameters

        Parameters
        ----------
        dim : int
            dimensionality of bias
        value : float
            initial value of bias parameter
        istrainable : bool
            flag indicating whether the parameters are trainable or not

        Attributes
        ----------
        param : ndarray (dim,)
            bias parameter to be added
        deriv : ndarray (dim,)
            derivative of cost function with respect to the parameter
        """
        self.param = np.zeros(dim) + value
        self.deriv = np.zeros_like(self.param)
        self.istrainable = istrainable

    def forward(self, x, *args):
        """
        forward propagation
        x + b

        Parameters
        ----------
        x : ndarray (sample_size, dim)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim)
            x + param
        """
        return x + self.param

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
        if self.istrainable:
            axes = tuple([i for i in range(delta.ndim - 1)])
            self.deriv = np.sum(delta, axes)
        return delta
