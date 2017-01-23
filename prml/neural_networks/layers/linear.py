from .layer import Layer
from .biasadd import BiasAdd
from .matmul import MatMul


class Linear(Layer):
    """
    Linear transformation layer

    Parameters
    ----------
    dim_in : int
            dimensionality of input
    dim_out : int
        dimensionality of output
    std : float
        standard deviation of truncnorm distribution for initializing parameter
    bias : float
        initial value of bias parameter
    istrainable : bool
        flag indicating trainable or not
    """

    def __new__(cls, dim_in, dim_out, std=1., bias=0., istrainable=True):
        """
        construct linear transformation layer

        Parameters
        ----------
        dim_in : int
            dimensionality of input
        dim_out : int
            dimensionality of output
        std : float
            standard deviation of truncnorm distribution for initializing parameter
        bias : float
            initial value of bias parameter
        istrainable : bool
            flag indicating trainable or not

        Returns
        -------
        matmul : MatMul
            Matrix multiplication layer
        add : Add
            Bias addition layer
        """
        return (
            MatMul(dim_in, dim_out, std=std, istrainable=istrainable),
            BiasAdd(dim_out, value=bias, istrainable=istrainable))
