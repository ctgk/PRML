import numpy as np
from prml.random.random import RandomVariable


class Wishart(RandomVariable):
    """
    Wishart distribution
    p(L|W(scale),nu(dof))
    = |L|^(nu - D - 1)/2 exp(-0.5 * Tr(W^-1 L)) / const.
    """

    def __init__(self, scale, dof):
        """
        construct Wishart distribution

        Parameters
        ----------
        scale : (ndim, ndim) np.ndarray
            scale matrix
        dof : int or float
            degree of freedom
        """
        assert isinstance(dof, (int, float))
        assert isinstance(scale, np.ndarray)
        self.scale = scale
        self.dof = dof

    def __repr__(self):
        return "Wishart(dof={0},\nscale=\n{0}\n)".format(self.dof, self.scale)

    @property
    def mean(self):
        return self.dof * self.scale
