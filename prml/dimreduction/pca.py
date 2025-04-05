import numpy as np


class PCA(object):

    def __init__(self, n_components):
        """
        construct principal component analysis

        Parameters
        ----------
        n_components : int
            number of components
        """
        assert isinstance(n_components, int)
        self.n_components = n_components

    def fit(self, x, method="eigen", iter_max=100):
        r"""Maximum likelihood estimate of pca parameters.

        x ~ \int_z N(x|Wz+mu,sigma^2)N(z|0,I)dz

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data
        method : str
            method to estimate the parameters
            ["eigen", "em"]
        iter_max : int
            maximum number of iterations for em algorithm

        Attributes
        ----------
        mean : (n_features,) ndarray
            sample mean of the data
        W : (n_features, n_components) ndarray
            projection matrix
        var : float
            variance of observation noise
        C : (n_features, n_features) ndarray
            variance of the marginal dist N(x|mean,C)
        Cinv : (n_features, n_features) ndarray
            precision of the marginal dist N(x|mean, C)
        """
        method_list = ["eigen", "em"]
        if method not in method_list:
            print("availabel methods are {}".format(method_list))
        self.mean = np.mean(x, axis=0)
        getattr(self, method)(x - self.mean, iter_max)

    def eigen(self, x, *arg):
        sample_size, n_features = x.shape
        if sample_size >= n_features:
            cov = np.cov(x, rowvar=False)
            values, vectors = np.linalg.eigh(cov)
            index = n_features - self.n_components
        else:
            cov = np.cov(x)
            values, vectors = np.linalg.eigh(cov)
            vectors = (x.T @ vectors) / np.sqrt(sample_size * values)
            index = sample_size - self.n_components
        self.eye = np.eye(self.n_components)
        if index == 0:
            self.var = 0
        else:
            self.var = np.mean(values[:index])

        self.W = vectors[:, index:].dot(np.sqrt(np.diag(values[index:]) - self.var * self.eye))
        self.__M = self.W.T @ self.W + self.var * self.eye
        self.C = self.W @ self.W.T + self.var * np.eye(n_features)
        if index == 0:
            self.Cinv = np.linalg.inv(self.C)
        else:
            self.Cinv = np.eye(n_features) / np.sqrt(self.var) - self.W @ np.linalg.inv(self.__M) @ self.W.T / self.var

    def em(self, x, iter_max):
        self.eye = np.eye(self.n_components)
        self.W = np.eye(np.size(x, 1), self.n_components)
        self.var = 1.
        for i in range(iter_max):
            W = np.copy(self.W)
            stats = self._expectation(x)
            self._maximization(x, *stats)
            if np.allclose(W, self.W):
                break
        self.C = self.W @ self.W.T + self.var * np.eye(np.size(x, 1))
        self.Cinv = np.linalg.inv(self.C)

    def _expectation(self, x):
        self.__M = self.W.T @ self.W + self.var * self.eye
        Minv = np.linalg.inv(self.__M)
        Ez = x @ self.W @ Minv
        Ezz = self.var * Minv + Ez[:, :, None] * Ez[:, None, :]
        return Ez, Ezz

    def _maximization(self, x, Ez, Ezz):
        self.W = x.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis=0))
        self.var = np.mean(
            np.mean(x ** 2, axis=1)
            - 2 * np.mean(Ez @ self.W.T * x, axis=1)
            + np.trace((Ezz @ self.W.T @ self.W).T) / np.size(x, 1))

    def transform(self, x):
        """
        project input data into latent space
        p(Z|x) = N(Z|(x-mu)WMinv, sigma^-2M)

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        Z : (sample_size, n_components) ndarray
            projected input data
        """
        return np.linalg.solve(self.__M, ((x - self.mean) @ self.W).T).T

    def fit_transform(self, x, method="eigen"):
        """
        perform pca and whiten the input data

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        Z : (sample_size, n_components) ndarray
            projected input data
        """
        self.fit(x, method)
        return self.transform(x)

    def proba(self, x):
        """
        the marginal distribution of the observed variable

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input data

        Returns
        -------
        p : (sample_size,) ndarray
            value of the marginal distribution
        """
        d = x - self.mean
        return (
            np.exp(-0.5 * np.sum(d @ self.Cinv * d, axis=-1))
            / np.sqrt(np.linalg.det(self.C))
            / np.power(2 * np.pi, 0.5 * np.size(x, 1)))
