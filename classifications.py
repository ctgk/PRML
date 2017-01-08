import numpy as np
import distributions


class LeastSquares(object):

    def fit(self, X, t):
        """
        perform least squares algorithm for classification

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels

        Returns
        -------
        W : ndarray (n_features, n_classes)
            parameter estimated by least squares alg.
        """
        T = np.eye(int(np.max(t)) + 1)[t]
        self.W = np.linalg.pinv(X) @ T

    def predict(self, X):
        """
        classify input data

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size)
            class labels
        """
        return np.argmax(X @ self.W, axis=1)


class LinearDiscriminantAnalysis(object):

    def fit(self, X, t):
        """
        estimate decision boundary by Linear Discriminant Analysis

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels

        Attributes
        ----------
        w : ndarray (n_features,)
            normal vector of hyperplane defining decision boundary
        threshold : float
            boundary value in projected space
        """
        assert np.max(t) + 1 == 2
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        cov_inclass = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)
        self.w = np.linalg.inv(cov_inclass) @ (m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=1e-10)
        g0 = distributions.Gaussian(X0 @ self.w)
        g1 = distributions.Gaussian(X1 @ self.w)
        a = g1.var - g0.var
        b = g0.var * g1.mean - g1.var * g0.mean
        c = (
            g1.var * g0.mean ** 2 - g0.var * g1.mean ** 2
            - g1.var * g0.var * np.log(g1.var / g0.var))
        self.threshold = (np.sqrt(b ** 2 - a * c) - b) / a

    def predict(self, X):
        """
        predict class labels

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        labels : ndarray (sample_size,)
            class labels
        """
        return (X @ self.w > self.threshold).astype(np.int)


class LogisticRegression(object):

    def __init__(self, alpha=0):
        """
        set precision parameter for prior distribution p(w|alpha)

        Parameters
        ----------
        alpha : float
            precision parameter for prior distribution
        """
        self.alpha = alpha

    def _sigmoid(self, a):
        return np.divide(1, 1 + np.exp(-a))

    def fit(self, X, t, iter_max=100):
        """
        Iterative reweighted least squares method to estimate parameter

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels
        iter_max : int
            number of maximum iterations

        Attributes
        ----------
        w : ndarray (n_features,)
            estimated parameter
        """
        self.w = np.zeros(np.size(X, 1))
        for i in range(iter_max):
            w = np.copy(self.w)
            y = self.predict_proba(X)
            grad = X.T @ (y - t) + self.alpha * w
            hessian = X.T @ np.diag(y * (1 - y)) @ X + self.alpha * np.eye(len(w))
            try:
                self.w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, self.w):
                break
        else:
            print("parameters may not have converged")

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(np.int)

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w)


class MultiLogisticRegression(object):

    def __init__(self, alpha=0.):
        """
        set precision parameter for prior distribution p(w|alpha)

        Parameters
        ----------
        alpha : float
            precision parameter for prior distribution
        """
        self.alpha = alpha

    def _softmax(self, a):
        """
        softmax function

        Parameters
        ----------
        a : ndarray (..., n_classes)
            activations

        Returns
        -------
        output : ndarray (...,)
            output of softmax function
        """
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, X, t, learning_rate=0.1, iter_max=100):
        """
        perform gradient descent algorithm to estimate weight parameter

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target class labels
        learning_rate : float
            step size of gradient descent update
        iter_max : int
            maximum number of iterations

        Attributes
        ----------
        w : ndarray (n_features, n_classes)
            estimated paramters
        """
        n_classes = np.max(t) + 1
        T = np.eye(n_classes)[t]
        self.w = np.zeros((np.size(X, 1), n_classes))
        for _ in range(iter_max):
            w = np.copy(self.w)
            y = self.predict_proba(X)
            grad = X.T @ (y - T) + self.alpha * w
            self.w -= learning_rate * grad
            if np.allclose(w, self.w):
                break
        else:
            print("parameter may not have converged")

    def predict(self, X):
        """
        predict class label of each input datum

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        labels : ndarray (sample_size,)
            predicted labels
        """
        return np.argmax(self.predict_proba(X), axis=-1)

    def predict_proba(self, X):
        """
        computer probability for each class

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data

        Returns
        -------
        y : ndarray (sample_size, n_classes)
            probability for each class
        """
        return self._softmax(X @ self.w)


class BayesianLogisticRegression(LogisticRegression):

    def fit(self, X, t, iter_max=100):
        super().fit(X, t, iter_max)
        y = self.predict_proba(X)
        hessian = X.T @ np.diag(y * (1 - y)) @ X + self.alpha * np.eye(len(self.w))
        self.w_cov = np.linalg.inv(hessian)

    def predict_dist(self, X):
        mu_a = X @ self.w
        var_a = np.sum(X @ self.w_cov * X, axis=1)
        return self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
