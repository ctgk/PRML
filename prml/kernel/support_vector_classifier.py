import numpy as np


class SupportVectorClassifier(object):

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

    def fit(self, X, t, learning_rate=0.1, decay_step=10000, decay_rate=0.9, min_lr=1e-5):
        """
        estimate decision boundary and its support vectors

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        t : (sample_size,) ndarray
            corresponding labels 1 or -1
        learning_rate : float
            update ratio of the lagrange multiplier
        decay_step : int
            number of iterations till decay
        decay_rate : float
            rate of learning rate decay
        min_lr : float
            minimum value of learning rate

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
        lr = learning_rate
        t2 = np.sum(np.square(t))
        if self.C == np.Inf:
            a = np.ones(len(t))
        else:
            a = np.zeros(len(t)) + self.C / 10
        Gram = self.kernel(X, X)
        H = t * t[:, None] * Gram
        while True:
            for i in range(decay_step):
                grad = 1 - H @ a
                a += lr * grad
                a -= (a @ t) * t / t2
                np.clip(a, 0, self.C, out=a)
            mask = a > 0
            self.X = X[mask]
            self.t = t[mask]
            self.a = a[mask]
            self.b = np.mean(
                self.t - np.sum(
                    self.a * self.t
                    * self.kernel(self.X, self.X),
                    axis=-1))
            if self.C == np.Inf:
                if np.allclose(self.distance(self.X) * self.t, 1, rtol=0.01, atol=0.01):
                    break
            else:
                if np.all(np.greater_equal(1.01, self.distance(self.X) * self.t)):
                    break
            if lr < min_lr:
                break
            lr *= decay_rate

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
