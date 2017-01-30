import numpy as np


class SupportVectorClassfier(object):

    def __init__(self, kernel, C=np.Inf):
        """
        construct support vector classifier

        Parameters
        ----------
        kernel : Kernel
            kernel function to compute inner products
        C : float
            penalty of misclassification
        """
        self.kernel = kernel
        self.C = C

    def fit(self, X, t, n_iter=100000, learning_rate=0.1, decay_step=1000, decay_rate=0.9):
        """
        estimate decision boundary and its support vectors

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        t : (sample_size,) ndarray
            corresponding labels 1 or -1
        n_iter : int
            number of iterations
        learning_rate : float
            update ratio of the lagrange multiplier
        decay_step : int
            steps to decay learning rate
        decay_rate : float
            rate of learning rate decay

        Attributes
        ----------
        a : (sample_size,) ndarray
            lagrange multiplier
        b : float
            bias parameter
        support_vector : (n_vector, n_features) ndarray
            support vectors of the boundary
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        self.X = X
        self.t = t
        t2 = np.sum(np.square(self.t))
        self.a = np.ones(X.shape[0])
        Gram = self.kernel(X, X)
        H = t * t[:, None] * Gram
        for i in range(n_iter):
            grad = 1 - H @ self.a
            self.a += learning_rate * grad
            self.a -= (self.a @ self.t) * self.t / t2
            np.clip(self.a, 0, self.C, out=self.a)
            if i % decay_step == 0:
                learning_rate *= decay_rate
        mask = self.a > 0
        self.X = self.X[mask]
        self.t = self.t[mask]
        self.a = self.a[mask]
        self.b = np.mean(
            self.t - np.sum(
                self.a * self.t
                * self.kernel(self.X, self.X),
                axis=-1))

    def lagrangian_function(self):
        return (
            np.sum(self.a)
            - self.a
            @ (self.t * self.t[:, None] * self.kernel(self.X, self.X))
            @ self.a)

    def predict(self, x):
        """
        predict labels of the input

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        label : (sample_size,) ndarray
            predicted labels
        """
        y = self.distance(x)
        label = np.sign(y)
        return label

    def distance(self, x):
        """
        calculate distance from the decision boundary

        Parameters
        ----------
        x : (sample_size, n_features) ndarray
            input

        Returns
        -------
        distance : (sample_size,) ndarray
            distance from the boundary
        """
        distance = np.sum(
            self.a * self.t
            * self.kernel(x, self.X),
            axis=-1) + self.b
        return distance
