from prml.nn.optimizer.optimizer import Optimizer


class Gradient(Optimizer):

    def __init__(self, parameter, learning_rate=1e-3):
        super().__init__(parameter, learning_rate)

    def update(self):
        for param in self.parameter.values():
            param.value += self.learning_rate * param.grad

