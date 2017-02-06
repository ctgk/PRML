import numpy as np


class UniformDistribution(object):
    """
    uniform distribution
    p(x|(a0,a1),(b0,b1)) = 1 / ((a1 - a0) * (b1 - b0))
    """
    def __init__(self, *arg):
        """
        construct uniform distribution

        Parameters
        ----------
        arg : tuple
            domain of non-zero probability
            eg. (1, 2), (1.5, 3)

        Attributes
        ----------
        ndim : int
            dimensionality
        value : float
            the value of non-zero probability
        """
        self.ndim = len(arg)
        self.value = 1.
        for tuple_ in arg:
            assert len(tuple_) == 2, len(tuple_)
            for n in tuple_:
                assert isinstance(n, int) or isinstance(n, float)
            self.value *= tuple_[1] - tuple_[0]
        self.value = 1 / self.value
        self.domain = arg

    def proba(self, X):
        """
        probability density function at the input

        Parameters
        ----------
        X : (sample_size, ndim) ndarray
            input data

        Returns
        -------
        p : (sample_size,) ndarray
            value of the probability density function
        """
        p = np.ones(len(X), dtype=np.bool)
        for i, segment in enumerate(self.domain):
            p *= segment[0] < X[:, i] < segment[1]
        p = p.astype(np.float) * self.value
        return p

    def draw(self, n=1):
        """
        draw random sample from this distribution

        Parameters
        ----------
        n : int
            number of samples to draw

        Returns
        -------
        sample : (n, ndim)
            generated sample
        """
        sample = []
        for segment in self.domain:
            sample.append(np.random.uniform(*segment, size=n))
        sample = np.asarray(sample).T
        assert sample.shape == (n, self.ndim), sample.shape
        return sample
