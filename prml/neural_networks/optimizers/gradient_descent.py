from .optimizer import Optimizer


class GradientDescentOptimizer(Optimizer):
    """
    gradient descent optimizer
    """

    def update(self):
        """
        update parameters of the neural network
        """
        self.increment_iteration()
        for layer in self.network.trainables:
            layer.param -= self.learning_rate * layer.deriv
