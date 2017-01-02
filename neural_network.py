import numpy as np
from scipy.stats import truncnorm


class Layer(object):

    def __init__(self, dim_input, dim_output, std=1., bias=0.):
        self.w = truncnorm(a=-2 * std, b=2 * std, scale=std).rvs((dim_input, dim_output))
        self.b = np.ones(dim_output) * bias

    def forward(self, X):
        self.input = X
        return X @ self.w + self.b

    def backprop(self, delta, learning_rate):
        w = np.copy(self.w)
        self.w -= learning_rate * self.input.T @ delta
        self.b -= learning_rate * np.sum(delta, axis=0)
        return delta @ w.T


class Sigmoid(object):

    def forward(self, X):
        self.output = np.divide(1, 1 + np.exp(-X))
        return self.output

    def backprop(self, delta, *arg):
        return self.output * (1 - self.output) * delta


class Tanh(object):

    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backprop(self, delta, *arg):
        return (1 - np.square(self.output)) * delta


class ReLU(object):

    def forward(self, X):
        self.output = X.clip(min=0)
        return self.output

    def backprop(self, delta, *arg):
        return (self.output > 0).astype(np.float) * delta


class SigmoidCrossEntropy(object):

    def forward(self, logits):
        return 1 / (1 + np.exp(-logits))

    def __call__(self, logits, targets):
        probs = self.forward(logits)
        p = np.clip(probs, 1e-10, 1 - 1e-10)
        return np.sum(-targets * np.log(p) - (1 - targets) * np.log(1 - p))

    def delta(self, logits, targets):
        probs = self.forward(logits)
        return probs - targets


class SoftmaxCrossEntropy(object):

    def forward(self, logits):
        a = np.exp(logits - np.max(logits, 1, keepdims=True))
        a /= np.sum(a, 1, keepdims=True)
        return a

    def __call__(self, logits, targets):
        probs = self.forward(logits)
        p = probs.clip(min=1e-10)
        return - np.sum(targets * np.log(p))

    def delta(self, logits, targets):
        probs = self.forward(logits)
        return probs - targets


class SumSquaresError(object):

    def forward(self, X):
        return X

    def __call__(self, X, targets):
        return 0.5 * np.sum((X - targets) ** 2)

    def delta(self, X, targets):
        return X - targets


class NeuralNetwork(object):

    def __init__(self, layers, cost_function):
        self.layers = layers
        self.layers.append(cost_function)
        self.cost_function = cost_function

    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def fit(self, X, t, learning_rate):
        for layer in self.layers[:-1]:
            X = layer.forward(X)

        delta = self.cost_function.delta(X, t)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backprop(delta, learning_rate)

    def cost(self, X, t):
        for layer in self.layers[:-1]:
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
        x_plus_e = X + e
        x_minus_e = X - e
        grad = (self.cost(x_plus_e, t) - self.cost(x_minus_e, t)) / (2 * eps)

        for layer in self.layers:
            X = layer.forward(X)
        delta = self.cost_function.delta(X, t)
        for layer in reversed(self.layers):
            delta = layer.backprop(delta, 0)

        print("===================================")
        print("checking gradient")
        print("finite difference", grad)
        print(" back propagation", delta[0, 0])
        print("above two gradients should be close")
        print("===================================")
