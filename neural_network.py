import numpy as np
from scipy.stats import truncnorm


class Layer(object):

    def __init__(self):
        """
        set flag indicating trainable or not

        Attributes
        ----------
        istrainable : bool
            flag indicating whether this layer is trainable or not
        """
        self.istrainable = False


class MatMul(Layer):
    """Matrix multiplication"""

    def __init__(self, dim_in, dim_out, std=1., alpha=0., istrainable=True):
        """
        initialize this layer

        Parameters
        ----------
        dim_in : int
            dimensionality of input
        dim_out : int
            dimensionality of output
        std : float
            standard deviation of truncnorm distribution for initializing parameter
        alpha : float
            precision parameter of prior distribution
        istrainable : bool
            flag indicating trainable or not

        Returns
        -------
        param : ndarray (dim_in, dim_out)
            coefficient to be matrix multiplied to the input
        deriv : ndarray (dim_in, dim_out)
            derivative of a cost function with respect to the paramter
        """
        self.param = truncnorm(
            a=-2 * std, b=2 * std, scale=std).rvs((dim_in, dim_out))
        self.deriv = np.zeros_like(self.param)
        self.alpha = alpha
        self.istrainable = istrainable

    def forward(self, X):
        """
        forward propagation
        X @ w

        Parameters
        ----------
        X : ndarray (sample_size, dim_in)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim_out)
            X @ w
        """
        self.input = X
        return X @ self.param

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
        delta_in = delta @ self.param.T
        if self.istrainable:
            self.deriv = self.input.T @ delta + self.alpha * self.param
        return delta_in


class Add(Layer):
    """Add bias"""

    def __init__(self, dim, value=0., alpha=0., istrainable=True):
        """
        initialize parameters

        Parameters
        ----------
        dim : int
            dimensionality of bias
        value : float
            initial value of bias parameter
        alpha : float
            precision parameter of prior distribution
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
        self.alpha = alpha
        self.istrainable = istrainable

    def forward(self, X):
        """
        forward propagation
        X + param

        Parameters
        ----------
        X : ndarray (sample_size, dim)
            input data

        Returns
        -------
        output : ndarray (sample_size, dim)
            X + param
        """
        return X + self.param

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
            self.deriv = np.sum(delta, axis=0) + self.alpha * self.param
        return delta


class Sigmoid(Layer):
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

    def backward(self, delta):
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


class Tanh(Layer):
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

    def backward(self, delta):
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


class ReLU(Layer):
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

    def backward(self, delta):
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

    def __init__(self, axis=-1):
        """
        set axis to take softmax function along

        Parameters
        ----------
        axis : int
            axis to take softmax along
            Default : -1
        """
        self.axis = axis

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

    def forward(self, logits):
        """
        softmax function along the given axis
        exp(logit_i) / sum_j{exp(logit_j)}

        Parameters
        ----------
        logits : ndarray
            input

        Returns
        -------
        a : ndarray
            softmax
        """
        a = np.exp(logits - np.max(logits, self.axis, keepdims=True))
        a /= np.sum(a, self.axis, keepdims=True)
        return a

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


class GaussianMixtureNLL(object):
    """Negative Log Likelihood of Mixture of Gaussian model"""

    def __init__(self, n_components):
        """
        set number of gaussian components

        Parameters
        ----------
        n_component : int
            number of gaussian components
        """
        self.n_components = n_components

    def __call__(self, X, targets):
        """
        Negative log likelihood of mixture of gaussian with given input

        Parameters
        ----------
        X : ndarray (sample_size, 3 * n_components)
            input
        targets : ndarray (sample_size, 1)
            corresponding target data

        Returns
        -------
        output : float
            negative log likelihood of mixture of gaussian
        """
        sigma, weight, mu = self.forward(X)
        gauss = self.gauss(mu, sigma, targets)
        return -np.sum(np.log(np.sum(weight * gauss, axis=1)))

    def gauss(self, mu, sigma, targets):
        """
        gauss function

        Parameters
        ----------
        mu : ndarray (sample_size, n_components)
            mean of each gaussian component
        sigma : ndarray (sample_size, n_components)
            standard deviation of each gaussian component
        targets : ndarray (sample_size, 1)
            corresponding target data

        Returns
        -------
        output : ndarray (sample_size, n_components)
            gaussian
        """
        return np.exp(-0.5 * (mu - targets) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))

    def forward(self, X):
        """
        compute parameters of mixture of gaussian model

        Parameters
        ----------
        X : ndarray (sample_size, 3 * n_components)
            input

        Returns
        -------
        sigma : ndaray (sample_size, n_components)
            standard deviation of each gaussian component
        weight : ndarray (sample_size, n_components)
            mixing coefficients of mixture of gaussian model
        mu : ndarray (sample_size, n_components)
            mean of each gaussian component
        """
        assert np.size(X, 1) == 3 * self.n_components
        X_sigma, X_weight, X_mu = np.split(X, [self.n_components, 2 * self.n_components], axis=1)
        sigma = np.exp(X_sigma)
        weight = np.exp(X_weight - np.max(X_weight, 1, keepdims=True))
        weight /= np.sum(weight, axis=1, keepdims=True)
        return sigma, weight, X_mu

    def backward(self, X, targets):
        """
        compute input errors

        Parameters
        ----------
        X : ndarray (sample_size, 3 * n_components)
            input
        targets : ndarray (sample_size, 1)
            corresponding target data

        Returns
        -------
        delta : ndarray (sample_size, 3 * n_components)
            input errors
        """
        sigma, weight, mu = self.forward(X)
        var = np.square(sigma)
        gamma = weight * self.gauss(mu, sigma, targets)
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        delta_mu = gamma * (mu - targets) / var
        delta_sigma = gamma * (1 - (mu - targets) ** 2 / var)
        delta_weight = weight - gamma
        delta = np.hstack([delta_sigma, delta_weight, delta_mu])
        return delta


class GradientDescent(object):

    def __init__(self, nn, learning_rate):
        """
        set neural network to be optimized

        Parameters
        ----------
        nn : NeuralNetwork
            neural network to be optimized
        learning_rate : float
            coefficient to be multiplied with gradient
        """
        self.nn = nn
        self.learning_rate = learning_rate

    def update(self, X, t):
        """
        update parameters of the neural network

        Parameters
        ----------
        X : ndarray (sample_size, ...)
            input
        t : ndarray (sample_size, ...)
            corresponding target data
        """
        for layer in self.nn.layers:
            X = layer.forward(X)

        delta = self.nn.cost_function.backward(X, t)
        for layer in reversed(self.nn.layers):
            delta = layer.backward(delta)
            if layer.istrainable:
                layer.param -= self.learning_rate * layer.deriv


class GradientDescentExponentialDecay(object):

    def __init__(self, nn, initial_learning_rate, decay_step, decay_rate):
        """
        set neural network and several parameters

        Parameters
        ----------
        nn : NeuralNetwork
            neural network to be optimized
        initial_learning_rate : float
            initial value of learning rate
        decay_step : int
            step size to decay learning rate
        decay_rate : float
            decay rate of learningn rate

        Attributes
        ----------
        iter_count : int
            number of times updated
        """
        self.nn = nn
        self.learning_rate = initial_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.iter_count = 0

    def update(self, X, t):
        """
        update parameters of the neural network and update learning rate

        Parameters
        ----------
        X : ndarray (sample_size, ...)
            input
        t : ndarray (sample_size, ...)
            corresponding target data
        """
        for layer in self.nn.layers:
            X = layer.forward(X)

        delta = self.nn.cost_function.backward(X, t)
        for layer in reversed(self.nn.layers):
            delta = layer.backward(delta)
            if layer.istrainable:
                layer.param -= self.learning_rate * layer.deriv

        self.iter_count += 1
        if self.iter_count % self.decay_step == 0:
            self.learning_rate *= self.decay_rate


class ConjugateGradientDescent(object):

    def __init__(self, nn, learning_rate):
        """
        set neural network to be optimized

        Parameters
        ----------
        nn : NeuralNetwork
            neural network to be optimized
        learning_rate : float
            updation coefficient

        Attributes
        ----------
        direction : dict
            updation direction for each parameter
        """
        self.nn = nn
        self.learning_rate = learning_rate
        self.direction = 0
        total_size = 0
        for layer in self.nn.layers:
            if layer.istrainable:
                total_size += layer.param.size
        self.derivatives = np.zeros(total_size)

    def update(self, X, t):
        """
        update parameters of neural network using conjugate gradient method

        Parameters
        ----------
        X : ndarray (sample_size, ...)
            input data
        t : ndarray (sample_size, ...)
            corresponding target data
        """
        for layer in self.nn.layers:
            X = layer.forward(X)

        deriv_old = self.derivatives
        self.derivatives = np.array([])
        sizes = []
        delta = self.nn.cost_function.backward(X, t)
        for layer in reversed(self.nn.layers):
            delta = layer.backward(delta)
            if layer.istrainable:
                self.derivatives = np.append(self.derivatives, layer.deriv.ravel())
                sizes.append(layer.param.size)
            else:
                sizes.append(0)
        indices = np.cumsum(sizes)

        b = np.sum(self.derivatives ** 2) / np.sum(deriv_old ** 2).clip(min=1e-10)
        self.direction = b * self.direction - self.derivatives
        directions = np.split(self.direction, indices)

        for layer, d in zip(reversed(self.nn.layers), directions):
            if layer.istrainable:
                layer.param += self.learning_rate * np.reshape(d, layer.param.shape)


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

    def update(self, X, t, learning_rate):
        for layer in self.layers:
            X = layer.forward(X)

        delta = self.cost_function.backward(X, t)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            if layer.istrainable:
                layer.update(learning_rate)

    def cost(self, X, t):
        for layer in self.layers:
            X = layer.forward(X)
        return self.cost_function(X, t)

    def check_implementation(self, X, t, eps=1e-6):
        e = np.zeros_like(X)
        e[:, 0] += eps
        grad = (self.cost(X + e, t) - self.cost(X - e, t)) / (2 * eps)

        for layer in self.layers:
            X = layer.forward(X)
        delta = self.cost_function.backward(X, t)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        print("===============================================")
        print("finite difference:", grad)
        print("back propagation :", delta[0, 0])
        print("The two values should be approximately the same")
        print("===============================================")
