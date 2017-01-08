import numpy as np
from scipy.stats import truncnorm


class Linear(object):

    def __init__(self, dim_in, dim_out, std=1., bias=0., alpha=0.):
        """
        initialize parameters

        Parameters
        ----------
        dim_in : int
            dimensionality of input
        dim_out : int
            dimensionality of output
        std : float
            standard deviation of truncnorm distribution
        bias : float
            initial value of bias parameter
        alpha : float
            precision parameter of prior distribution

        Attributes
        ----------
        w : ndarray (dim_in, dim_out)
            coefficients to be multiplied to the input
        delta_w : ndarray (dim_in, dim_out)
            derivative of cost function with respect to w
        b : ndarray (dim_out,)
            bias parameter to be added
        delta_b : ndarray (dim_out,)
            derivative of cost function with respect to b
        """
        self.w = truncnorm(
            a=-2 * std, b=2 * std, scale=std).rvs((dim_in, dim_out))
        self.b = np.ones(dim_out) * bias
        self.delta_w = np.zeros_like(self.w)
        self.delta_b = np.zeros_like(self.b)
        self.alpha = alpha

    def forward(self, X):
        """
        forward propagation
        X @ w + b

        Parameters
        ----------
        X : ndarray (sample_size, dim_in)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim_out)
            X @ w + b
        """
        self.input = X
        return X @ self.w + self.b

    def backward(self, delta, learning_rate):
        """
        backpropagation of errors

        Parameters
        ----------
        delta : ndarray (sample_size, dim_out)
            output error
        learning_rate : float
            for updating parameters

        Returns
        -------
        delta_in : ndarray (sample_size, dim_in)
            input error
        """
        delta_in = delta @ self.w.T
        self.delta_w = self.input.T @ delta + self.alpha * self.w
        self.delta_b = np.sum(delta, axis=0) + self.alpha * self.b
        self.w -= learning_rate * self.delta_w
        self.b -= learning_rate * self.delta_b
        return delta_in


class Sigmoid(object):
    """Logistic sigmoid function"""

    def forward(self, X):
        """
        element-wise logistic sigmoid function
        1 / (1 + exp(-X))

        Parameters
        ----------
        X : ndarray (sample_size, dim)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim)
            logist sigmoid of each element
        """
        self.output = np.divide(1, 1 + np.exp(-X))
        return self.output

    def backward(self, delta, *arg):
        """
        backpropagation of errors
        y = 1 / (1 + exp(-x))
        dy/dx = y * (1 - y)

        Parameters
        ----------
        delta : ndarray (sample_size, dim)
            output errors

        Returns
        -------
        delta_in : ndarray (sample_size, dim)
            input errors
        """
        return self.output * (1 - self.output) * delta


class Tanh(object):
    """tanh"""

    def forward(self, X):
        """
        element-wise tanh function

        Parameters
        ----------
        X : ndarray (sample_size, dim)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim)
            tanh of each element
        """
        self.output = np.tanh(X)
        return self.output

    def backward(self, delta, *arg):
        """
        backpropagation of errors
        y = tanh(x)
        dy/dx = 1 - y^2

        Parameters
        ----------
        delta : ndarray (sample_size, dim)
            output errors

        Returns
        -------
        delta_in : ndarray (sample_size, dim)
            input errors
        """
        return (1 - np.square(self.output)) * delta


class ReLU(object):
    """Rectified Linear Unit"""

    def forward(self, X):
        """
        element-wise rectified linear function
        max(X, 0)

        Parameters
        ----------
        X : ndarray (sample_size, dim)
            input

        Returns
        -------
        output : ndarray (sample_size, dim)
            rectified linear of each element
        """
        self.output = X.clip(min=0)
        return self.output

    def backward(self, delta, *arg):
        """
        backpropation of errors
        y = max(x, 0)
        dy/dx = 1 if x > 0 else 0

        Parameters
        ----------
        delta : ndarray (sample_size, dim)
            output errors

        Returns
        -------
        delta_in : ndarray (sample_size, dim)
            input errors
        """
        return (self.output > 0).astype(np.float) * delta


class SigmoidCrossEntropy(object):

    def forward(self, logits):
        """
        element-wise logistic sigmoid function
        1 / (1 + exp(-logits))

        Parameters
        ----------
        logits : ndarray
            input

        Returns
        -------
        output : ndarray
            logistic sigmoid of each element
        """
        return 1 / (1 + np.exp(-logits))

    def __call__(self, logits, targets):
        """
        cross entropy between target and sigmoid logit
        sum_i{-t_i*log(p_i)}

        Parameters
        ----------
        logits : ndarray
            inputs
        targets : ndarray
            target data

        Returns
        -------
        output : float
            sum of cross entropies
        """
        probs = self.forward(logits)
        p = np.clip(probs, 1e-10, 1 - 1e-10)
        return np.sum(-targets * np.log(p) - (1 - targets) * np.log(1 - p))

    def backward(self, logits, targets):
        """
        compute derivative with respect to the input

        Parameters
        ----------
        logits : ndarray
            input logits
        targets : ndarray
            target data

        Returns
        -------
        delta : ndarray
            input errors
        """
        probs = self.forward(logits)
        return probs - targets


class SoftmaxCrossEntropy(object):

    def forward(self, logits, axis=-1):
        """
        softmax function along the given axis
        exp(logit_i) / sum_j{exp(logit_j)}

        Parameters
        ----------
        logits : ndarray
            input
        axis : int
            axis to compute softmax along

        Returns
        -------
        a : ndarray
            softmax
        """
        a = np.exp(logits - np.max(logits, axis, keepdims=True))
        a /= np.sum(a, axis, keepdims=True)
        return a

    def __call__(self, logits, targets):
        """
        cross entropy between softmax logits and targets

        Parameters
        ----------
        logits : ndarray
            input
        targets : ndarray
            target data

        Returns
        -------
        output : float
            sum of cross entropies
        """
        probs = self.forward(logits)
        p = probs.clip(min=1e-10)
        return - np.sum(targets * np.log(p))

    def backward(self, logits, targets):
        """
        compute input errors

        Parameters
        ----------
        logits : ndarray
            input
        targets : ndarray
            target data

        Returns
        -------
        delta : ndarray
            input errors
        """
        probs = self.forward(logits)
        return probs - targets


class SumSquaresError(object):

    def forward(self, X):
        """
        identity function

        Parameters
        ----------
        X : ndarray
            input

        Returns
        -------
        output : ndarray
            identity of input
        """
        return X

    def __call__(self, X, targets):
        """
        sum of squared errors
        0.5 * ||X - targets||^2

        Parameters
        ----------
        X : ndarray
            input
        targets : ndarray
            corresponding target data

        Returns
        -------
        error : float
            sum of squared errors
        """
        return 0.5 * np.sum((X - targets) ** 2)

    def backward(self, X, targets):
        """
        compute input errors

        Parameters
        ----------
        X : ndarray
            input
        targets : ndarray
            corresponding target data

        Returns
        -------
        delta : ndarray
            input errors
        """
        return X - targets


class GaussianMixture(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def __call__(self, X, targets):
        sigma, weight, mu = self.forward(X)
        gauss = self.gauss(mu, sigma, targets)
        return -np.sum(np.log(np.sum(weight * gauss, axis=1)))

    def forward(self, X):
        assert np.size(X, 1) == 3 * self.n_components
        X_sigma, X_weight, X_mu = np.split(X, [self.n_components, 2 * self.n_components], axis=1)
        sigma = np.exp(X_sigma)
        weight = np.exp(X_weight - np.max(X_weight, 1, keepdims=True))
        weight /= np.sum(weight, axis=1, keepdims=True)
        return sigma, weight, X_mu

    def gauss(self, mu, sigma, targets):
        return np.exp(-0.5 * (mu - targets) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))

    def backward(self, X, targets):
        sigma, weight, mu = self.forward(X)
        var = np.square(sigma)
        gamma = weight * self.gauss(mu, sigma, targets)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        delta_mu = gamma * (mu - targets) / var
        delta_sigma = gamma * (1 - (mu - targets) ** 2 / var)
        delta_weight = weight - gamma
        delta = np.hstack([delta_sigma, delta_weight, delta_mu])
        return delta


class NeuralNetwork(object):

    def __init__(self, layers, cost_function):
        """
        define architecture and cost function

        Parameters
        ----------
        layers : list
            list of layer
        cost_function
            cost function to be minimized
        """
        self.layers = layers
        self.cost_function = cost_function

    def forward(self, X, n=None):
        """
        forward propagaton

        Parameters
        ----------
        X : ndarray
            input data
        n : int
            n-th layer's output will be returned
            Default : None

        Returns
        -------
        output : ndarray
            output of this network
        """
        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
            if i == n:
                return X
        return self.cost_function.forward(X)

    def fit(self, X, t, learning_rate):
        for layer in self.layers:
            X = layer.forward(X)

        delta = self.cost_function.backward(X, t)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)

    def cost(self, X, t):
        for layer in self.layers:
            X = layer.forward(X)
        return self.cost_function(X, t)

    def check_implementation(self, X=None, t=None, eps=1e-6):
        if X is None:
            X = np.array([[0.5 for _ in range(np.size(self.layers[0].w, 0))]])
        if t is None:
            t = np.zeros((1, np.size(self.layers[-1].w, 1)))
            t[0, 0] = 1.

        e = np.zeros_like(X)
        e[:, 0] += eps
        grad = (self.cost(X + e, t) - self.cost(X - e, t)) / (2 * eps)

        for layer in self.layers:
            X = layer.forward(X)
        delta = self.cost_function.backward(X, t)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, 0)

        print("===============================================")
        print("finite difference:", grad)
        print("back propagation :", delta[0, 0])
        print("The two values should be approximately the same")
        print("===============================================")
