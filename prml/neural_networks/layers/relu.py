from .layer import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit

    y = max(x, 0)
    """

    def forward(self, x, training=False):
        """
        element-wise rectified linear transformation

        Parameters
        ----------
        x : ndarray
            input

        Returns
        -------
        output : ndarray
            rectified linear of each element
        """
        if training:
            self.output = x.clip(min=0)
            return self.output
        else:
            return x.clip(min=0)

    def backward(self, delta):
        """
        backpropagation of errors

        Parameters
        ----------
        delta : ndarray
            output errors

        Returns
        -------
        delta_in : ndarray
            input errors
        """
        return (self.output > 0) * delta
