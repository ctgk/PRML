from prml.nn.function import Function


class ReLU(Function):

    @staticmethod
    def _forward(x):
        return x.clip(min=0)

    @staticmethod
    def _backward(delta, x):
        return delta * (x > 0)


def relu(x):
    return ReLU().forward(x)
