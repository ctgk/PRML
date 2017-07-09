from prml.nn.optimizer.optimizer import Optimizer


class GradientDescent(Optimizer):
    """
    gradient descent optimizer

    param -= learning_rate * gradient
    """

    def update(self):
        """
        update parameters of the neural network
        """
        self.increment_iteration()
        for param in self.params.values():
            param.value -= self.learning_rate * param.grad
