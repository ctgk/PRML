class Optimizer(object):

    def __init__(self, parameter: dict, learning_rate: float):
        if isinstance(parameter, list):
            self.parameter = {f"parameter{i}" : param for i, param in enumerate(parameter)}
        elif isinstance(parameter, dict):
            self.parameter = parameter
        self.learning_rate = learning_rate
        self.iter_count = 0

    def increment_iter_count(self):
        self.iter_count += 1

    def minimize(self, loss):
        if self.learning_rate > 0:
            self.learning_rate *= -1
        self.optimize(loss)

    def maximize(self, score):
        if self.learning_rate < 0:
            self.learning_rate *= -1
        self.optimize(score)

    def optimize(self, array):
        self.increment_iter_count()
        array.backward()
        self.update()

    def update(self):
        raise NotImplementedError
